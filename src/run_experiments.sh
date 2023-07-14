#!/bin/bash
# train_model.sh

# set the environment variables
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids


python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.2

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.2


python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \


python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.2

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.2

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --val_metric val_mcc \
        --dropout_prob 0.2

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_seg_and_raw \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --val_metric val_mcc \
        --dropout_prob 0.2