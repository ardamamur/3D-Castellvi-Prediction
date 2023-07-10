#!/bin/bash
# train_model.sh

# set the environment variable
export PYTHONWARNINGS="ignore"



python3 eval.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 8 \
        --learning_rate 0.00001 \
        --weight_decay 0.0001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.4  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --accumulate_grad_batches 8 \
        --version_no 2
