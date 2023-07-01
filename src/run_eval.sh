#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"



python3 eval.py --use_seg --use_bin_seg --use_zero_out --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 5
python3 eval.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 9
python3 eval.py --use_seg --use_bin_seg --use_zero_out --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 10
