#!/bin/bash
# train_model.sh

# set the environment variables
export PYTHONWARNINGS="ignore"


python3 train.py \
        --flip_all \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --dropout_prob 0.3 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --dropout_prob 0.1 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --elastic_transform \
        --dropout_prob 0.1 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --dropout_prob 0.3 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --dropout_prob 0.1 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --elastic_transform \
        --dropout_prob 0.1 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 1 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --elastic_transform \
        --dropout_prob 0.3 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 20 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --elastic_transform \
        --dropout_prob 0.3 \

python3 train.py \
        --flip_all \
        --use_zero_out \
        --use_seg \
        --use_seg_binary \
        --weighted_sample \
        --n_epochs 70 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 40 \
        --translate_range 0.4  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --accumulate_grad_batches 4 \
        --num_workers 8 \
        --elastic_transform \
        --dropout_prob 0.3 \