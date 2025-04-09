import sys, os
import time


from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS, print_memory_usage

cnt = 0
def my_print(s):
    global cnt
    cnt += 1
    print(s, end='', flush=True)
    if cnt % 50 == 0:
        print_memory_usage("\n")

# Model setup
model_path = "/data/models/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth"
strategy = "cpu fp16"
model = RWKV(model=model_path, strategy=strategy, verbose=True)

# Inference
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
ctx = "\nUniversity of Virginia is"
#print(ctx, end='')

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

TOKEN_CNT = 1 
pipeline.generate(ctx, token_count=TOKEN_CNT, args=args, callback=my_print)
print('\n')
