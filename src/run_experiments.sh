#!/bin/bash
# train_model.sh

# set the environment variables
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids


# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_zero_out \
#         --weighted_sample \
#         --n_epochs 85 \
#         --batch_size 16 \
#         --learning_rate 0.0001 \
#         --weight_decay 0.001 \
#         --classification_type right_side \
#         --model densenet \
#         --num_classes 3 \
#         --rotate_range 40 \
#         --translate_range 0.6  \
#         --scale_range 0.9 1.1 \
#         --aug_prob 1 \
#         --num_workers 8 \
#         --elastic_transform \

# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_zero_out \
#         --weighted_sample \
#         --n_epochs 85 \
#         --batch_size 16 \
#         --learning_rate 0.00001 \
#         --weight_decay 0.001 \
#         --classification_type right_side \
#         --model densenet \
#         --num_classes 3 \
#         --rotate_range 40 \
#         --translate_range 0.6  \
#         --scale_range 0.9 1.1 \
#         --aug_prob 1 \
#         --num_workers 8 \
#         --elastic_transform \

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type both_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \