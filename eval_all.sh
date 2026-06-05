#!/bin/bash

# Prohibit each job from accessing the huggingface hub to avoid the rate limitter.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

parallel msub  -p $(date +%Y%m%d%H%M)-kvpacket-eval -c 4 -t 12h -m 256g -g 1 python run_eval.py \
         ::: $(find eval_config -name "*.json" | grep -v _default)

