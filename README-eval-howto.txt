== Prep
Clone this repo
create a conda environment called "rwkv", install needed packages (via pip).
IMPORTANT: use Python3.10

RWKV-LM/rwkv: own version of rwkv inference engine (CUDA optimized)

RWKV-LM/lm_eval: own version of lm_eval (RWKV-LM/lm_eval) good for cusotmization, code comment, etc
    (instead of the one from pip3. TBD modify it so it works with rwkv) 

other dependencies: 
pip3 install pycountry datasets sacrebleu sqlitedict scikit-learn transformers

=== Key inference code
RWKV-LM/rwkv/model.py 
    this file has comments. 
    for compressed classfication head, cf: line ~1700
    
=== Run a model via chat (sanity check
python3.10 src/test-rwkv-chat.py

=== Eval a model against lm benchmarks
python3.10 src/run_lm_eval.py
For details, see comments at the top of run_lm_eval.py
To change models/benchmarks, cf the code 

=== Submit eval jobs on Rivanna 

cd RWKV-LM/RWKV-v5/out
mkdir 01b-pretrain-x52

cd 01b-pretrain-x52
cp ../../template/*.sh . 

./submit-eval.sh
Results will be written to eval_log.txt, eval_log.png


=== Run our models (x59)
```
# load our model
model_path = /path/to/your/model.pth

# load clustered tokens
cls_path = /path/to/your/cls.npy

# enable quantization predictor
quant_bit = bit # 1/2/4/8 bit
quant_map = [threshold1, threshold2, ...] # len(quant_map) == # of layers

# enable mlp predictor
mlp_map = [threshold1, threshold2, ...]  # len(mlp_map) == # of layers

# ex1) 1-bit quantization for 0.1B & MLP predictors
quant_bit = 1
quant_map = [0.9] * 12
mlp_map = [0.7] * 12


model = RWKV(model=model_path, strategy="cuda fp16", verboase=isverbose,
             quant_bit=quant_bit, quant_map=quant_map,
             mlp_map=mlp_map,
             load_token_cls=cls_path)
```

Troubleshooting
===============
Q. I see 'NaN' on perplexity and a zero accuracy as follows:
```bash
{
    "lambada_openai": {
        "perplexity,none": NaN,
        "perplexity_stderr,none": NaN,
        "acc,none": 0.0,
        "acc_stderr,none": 0.0,
        "alias": "lambada_openai"
    }
}
```

A. This happens because of the wrong floating point. Load a model as `fp32` instead of `fp16`.
A possible reason could be the following comment: "In [state = kv + w * state] everything must be in fp32 because w can be very close to 1. So we can keep state and w in fp32, and convert kv to fp32."


```
# src/run_lm_eval.py
- rwkv_model = RWKV(model=model_path, strategy="cuda fp16", verboase=isverbose)
+ rwkv_model = RWKV(model=model_path, strategy="cuda fp32", verboase=isverbose)
```
