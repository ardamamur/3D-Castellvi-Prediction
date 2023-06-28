#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"

# Data Augmentation
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 22
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 23
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 24
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 25
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 26
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 27

# # Zeroing Out
# ## Segmentation Images
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 28
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 29
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 30
# ## CT Images
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 31
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 32
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 33


python3 eval_V2.py --use_seg --use_bin_seg --use_zero_out --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 10