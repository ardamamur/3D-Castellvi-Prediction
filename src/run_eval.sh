#!/bin/bash
# train_model.sh

# set the environment variable
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids


python3 eval.py \
        --eval_type test \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --classification_type both_side \
        --model densenet \
        --num_classes 3 \
        --version_no 0
