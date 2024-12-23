'''
test rwkv inference engine
cf: https://pypi.org/project/rwkv/

speed benchmark res - see of file
full res: 
https://myuva.sharepoint.com/:x:/r/sites/XSEL-RWKV/Shared%20Documents/RWKV/results_rwkv.xlsx?d=wbf0bd61c5429469a8c039df4d8d4f46a&csf=1&web=1&e=0dyjUv
'''
import sys, os
import time

# run chat app on the inference engine (rwkv), check for sanity 

# xzl: use our own version of lm_eval, rwkv

home_dir = os.environ.get('HOME')
if home_dir == None: 
    home_dir = "/home/xl6yq"  # guessed
home_dir += "/"

sys.path.append(home_dir + 'workspace-rwkv/RWKV-LM')
if os.environ.get("RWKV_JIT_ON") != '0':
    os.environ["RWKV_JIT_ON"] = '1'

if os.environ.get('RWKV_CUDA_ON') != '0':
    os.environ["RWKV_CUDA_ON"] = '1' #default

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS, print_memory_usage
from rwkv.arm_plat import is_amd_cpu

import os


# rva
# model_path='/scratch/xl6yq/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096'

# official
# model_path='/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096' # official, NB it's v1
# model_path='/data/models/pi-deployment/RWKV-5-World-0.4B-v2-20231113-ctx4096'

# .1B 16x, deeply compressed 
# model_path='/data/models/01b-pre-x59-16x-901'

#v5.9
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-init'   #unmodified model,  pretrained by us 
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-25'
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01B-relu-diag-pretrain/rwkv-35'

# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run1/rwkv-7'  # old
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/rwkv-init'
# model_path='/data-xsel02/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run2/rwkv-24'  #Only head.l1 tuned

# model_path='/data/models/0.1b-pre-x59-16x-1451'
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-pretrain-x59/from-hpc/rwkv-976'

# model_path='/data/models/pi-deployment/01b-pre-x52-1455'        # works on opi0
# model_path='/data/models/pi-deployment/01b-pre-x58-512'

#model_path='/data/models/pi-deployment/01b-pre-x52-1455_fp16i8'     # can directly load quant model like this. cf "conversion" below
# model_path='/data/models/pi-deployment/01b-pre-x59-976'
# model_path='/data/models/pi-deployment/04b-tunefull-x58-562'   # works on opi0
# model_path='/data/models/pi-deployment/04b-pre-x59-2405'

model_path='/data/models/pi-deployment/1b5-pre-x59-929'
# model_path='/data/models/pi-deployment/1b5-pre-x59-929_fp16i8'
# model_path='/data/models/pi-deployment/01b-pre-x59-CLS-TEST'

# has mlp, cls (need to pass in hyperparam)
# model_path='/data/models/orin-deployment/01b-x59'
# model_path='/data/models/orin-deployment/04b-x59'

# #Only head.l1 tuned. KL loss (good
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'

#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-cls-mine/run3-KL-loss/rwkv-43'
#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-pre-x59-8x-cls/from-hpc/rwkv-1366'
#model_path='/data/home/bfr4xr/RWKV-LM/RWKV-v5/out/01b-pre-x59-8x-cls/from-hpc/0.1b-official'
# only head.l1fc1, head.l1fc2 (MLP) trained. KL loss
#   very bad
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run5-KL-loss-MLP-KaimingInit/rwkv-230'
#   very bad
# model_path='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/run4-KL-loss-MLP/rwkv-40'


print(f'Loading model - {model_path}')

# xzl: for strategy, cf: https://pypi.org/project/rwkv/ for more ex
#
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32

if os.environ["RWKV_CUDA_ON"] == '1':
    strategy='cuda fp16'
    # strategy='cuda fp16i8',
else:
    if is_amd_cpu():
        strategy='cpu fp32'  # amd cpu lacks hard fp16...
    else:
         strategy='cpu fp16'
        #strategy='cpu fp16i8'       # quant

# use below to quantize model & save
if False: 
    strategy_token = strategy.split()[1]
    basename, extension = os.path.splitext(os.path.basename(model_path))
    save_path = os.path.join(os.path.dirname(model_path), f"{basename}_{strategy_token}{extension}")
    print(f'Save path: {save_path}')
    model = RWKV(model=model_path, strategy=strategy, verbose=True, convert_and_save_and_exit=save_path)
    sys.exit(0)

print_memory_usage("before model build")

t0 = time.time()
model = RWKV(model=model_path, 
             strategy=strategy, 
             verbose=True)
#              head_K=200, load_token_cls='/data/home/xl6yq/workspace-rwkv/RWKV-LM/RWKV-v5/out/01b-cls-mine/from-hpc/rwkv-823-cls.npy')

print_memory_usage("before pipeline build")

pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# ex prompt from paper: https://arxiv.org/pdf/2305.07759
# ctx = "\nWhat is the sum of 123 and 456"
# ctx = "\nElon Musk has"
ctx = "\nUniversity of Virginia is"
# ctx = "\nAlice was so tired when she got back home so she went"
# ctx = "\nLily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked"
# ctx = "\nOnce upon a time there was a little girl named Lucy"
print(ctx, end='')

cnt=0

def my_print(s):
    global cnt
    cnt += 1
    print(s, end='', flush=True)
    if cnt % 50 == 0:
        print_memory_usage("\n")

t1 = time.time()

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

print_memory_usage("before generate")

TOKEN_CNT = 100 
pipeline.generate(ctx, token_count=TOKEN_CNT, args=args, callback=my_print)
print('\n')

t2 = time.time()

print(f"model build: {(t1-t0):.2f} sec, exec {TOKEN_CNT} tokens in {(t2-t1):.2f} sec, {TOKEN_CNT/(t2-t1):.2f} tok/sec")

if model.stat_runs != 0:
    print(f"stats: runs: {model.stat_runs} \
        cls/run {model.stat_loaded_cls/model.stat_runs:.2f} \
        tokens/run {model.stat_loaded_tokens/model.stat_runs/65535:.2f}")
      
'''
# xzl: what are thsse for??? demo cut a long prompt into pieces and feed??
out, state = model.forward([187, 510, 1563, 310, 247], None)
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
print('\n')
'''

'''
speed test 
(careful: vscode-server will take quite some cpu time)

rpi5 (supports fp16 in neon)
                                tok/sec                     mem(inference)
x52     01b-pre-x52-1455        15.3                
    quant fp16i8 v3                       ~12 tok/sec  
            (reason: openmp multithreading straggler?. occassionally, mm8 takes 3x-4x longer to finish)
    quant fp16i8 v4                       ~10 tok/sec  
            (even for M=N=768, 2t is still benefitical; better than 1t)
                                 
x59     01b-pre-x59-976         15.02 tok/s   mem=1GB 
        fp16i8 v3               11 tok/s      mem=.9GB

                               tok/s  mem(inference)
04b official (x52)             6.62    

04b-pre-x59-2405           6.86 tok/s   mem=3.5G             
    quant fp16i8 v3            3.45 tok/s   mem=1.3G
    quant fp32i8 v3            3.00 tok/s   mem=1.3G

1b5  1b5-pre-x59-929   (on rpi5 8GB RAM) 3 tok/s        mem=4G (??
     quant fp16i8 v3            1.95 toks/sec    mem=3.5G
     quant fp32i8 v3            1.1 toks/sec                mem=3.5G

--------------
rpi4  (BCM2711, quad Cortex-A72, 1.5GHz, 8GB)
(only support fp32 in neon)
                                tok/sec
x52     01b-pre-x52-1455        5.1      
    quant fp16i8                       .4 (very slow)
x59     01b-pre-x59-976         3.1                                
    quant fp16i8                      .45 (very slow)

    1b5                          .26 (slow)

--------------
orange pi zero2w (Allwinner H618; quad CortexA53, 1.5GHz; 4GB DRAM)
                        tok/sec
01b-pre-x52-1455        4.96
01b-pre-x52-1455 fp16i8 (<1) # very slow, and mem saving is not significant?
01b-pre-x59-976         4.72
04b-tunefull-x58-562    2.42    
04b-pre-x59-860         2.30

(naked cpu temp reached ~70C)
board level power: 0.43A@5.25v

--------------
xsel02, amd cpu (only support hw fp32, fp16 VERY slow

FP32 
(NB: this is only svd. no cls, no sparsification
                                 tok/sec
01b official (x52)             


                               
04b official (x52)              44.72
    04b-tunefull-x58-562        43.58
    04b-pre-x59-2405            31.22 
    
'''
