#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"


python3 train.py \
        --flip_all \
        --use_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.4  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 


python3 train.py \
        --flip_all \
        --use_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.4  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 



python3 train.py \
        --flip_all \
        --use_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.4  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 


python3 train.py \
        --flip_all \
        --use_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 
