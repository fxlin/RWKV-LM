# RWKV-Lite: Deeply Compressed RWKV for Resource-Constrained Devices


(TBD)

## Training

### Set up environments for training
```
# Create a conda environment
conda create -n rwkv python=3.10

# Install python requirements
pip install -r requirements.txt
```

### SVD Training

Step 1.  create a workspace directory

```
mkdir -p RWKV-v5/out/04b-x58
```

Step 2. copy over all training scripts

```
cd RWKV-v5
cp template/*.sh out/04b-x58
```
Files to change: 

* `model-config.sh` to set model variant, # of layers, etc. 
* `run-train.sh` to change training hyperparams, e.g. learning rate, etc
* `run-eval.sh` to change evaluation hyperparams, etc

Step 3. prepare a dataset: See [Dataset](#dataset).

Step 4. initialize a model
```
cd {your-path}/RWKV-v5/out/04b-x58
bash prep.sh
```

Step 5. run a training script
```
# check the current dir
pwd 
{your-path}/RWKV-Lite/RWKV-v5/out/04b-x58

# run a script
bash run-train.sh
```

## Evaluation
Step 0. create a symbolic link in `src`
```
pwd 
~/RWKV-Lite
ln -s $(pwd)/rwkv RWKV-v5/src/
```

Step 1. install `lm_eval`

```
cd {your-path}/RWKV-Lite
bash scripts/install-lm-eval.sh
```

Step 2. run a script
```
cd out/04b-x58
bash run-eval.sh
```


## Inference


## Inference (Raspberry Pi 5)

## Dataset
### A toy example
Step 1. Get `minipile` dataset
```
bash RWKV-v5/getdata.sh
ls RWKV-v5/data
minipile.bin    minipile.idx
```

## Model information
| paramter size | # of layers | Embedding dim |
| ------------- | ----------- | ------------- |
| 0.1B          | 12          | 768           |
| 0.4B          | 24          | 1024          |
| 1.5B          | 24          | 2048          |
| 3B            | 32          | 2560          |
| 7B            | 32          | 4096          |

