#!/bin/bash
# an evaluation script for other transformer models

readarray -t models < <(grep -v '#' models.txt)

for m in "${models[@]}"; do
    echo $m
    #lm_eval --model hf \
    #    --model_args pretrained=$m,revision=step100000,dtype="float" \
    #    --tasks lambada_openai \
    #    --batch_size 8
done
