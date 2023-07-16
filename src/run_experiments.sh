#!/bin/bash
# train_model.sh

# set the environment variables
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids

############# Experiment 0 #############
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
#         --val_metric val_mcc \
#         --dropout_prob 0.2


############# Experiment 1 #############
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
#         --val_metric val_mcc \


############# Experiment 2 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --val_metric val_mcc \
#         --dropout_prob 0.2



############# Experiment 3 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --val_metric val_mcc \



############# Experiment 4 #############
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
#         --aug_prob 0.5 \
#         --num_workers 8 \
#         --elastic_transform \
#         --val_metric val_mcc \
#         --dropout_prob 0.2


############# Experiment 5 #############
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
#         --aug_prob 0.5 \
#         --num_workers 8 \
#         --elastic_transform \
#         --val_metric val_mcc \


############# Experiment 6 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --aug_prob 0.5 \
#         --num_workers 8 \
#         --elastic_transform \
#         --val_metric val_mcc \
#         --dropout_prob 0.2

############# Experiment 7 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --aug_prob 0.5 \
#         --num_workers 8 \
#         --elastic_transform \
#         --val_metric val_mcc \


############# Experiment 8 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --val_metric val_mcc \
#         --dropout_prob 0.2


############# Experiment 9 #############
# python3 train.py \
#         --flip_all \
#         --use_seg \
#         --use_bin_seg \
#         --use_seg_and_raw \
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
#         --val_metric val_mcc \
#         --dropout_prob 0.2


############# Experiment 10 #############
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
#         --classification_type both_side \
#         --model densenet \
#         --num_classes 3 \
#         --rotate_range 40 \
#         --translate_range 0.6  \
#         --scale_range 0.9 1.1 \
#         --aug_prob 1 \
#         --num_workers 8 \
#         --elastic_transform \
#         --val_metric val_mcc \


############# Experiment 11 #############
python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --use_seg_and_raw \
        --weighted_sample \
        --n_epochs 160 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.4


############# Experiment 12 #############
python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --use_seg_and_raw \
        --weighted_sample \
        --n_epochs 160 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 40 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.4


############# Experiment 13 #############
python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --use_seg_and_raw \
        --weighted_sample \
        --n_epochs 160 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.6


############# Experiment 14 #############
python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --use_seg_and_raw \
        --weighted_sample \
        --n_epochs 160 \
        --batch_size 16 \
        --learning_rate 0.00001 \
        --weight_decay 0.001 \
        --classification_type right_side_binary \
        --model densenet \
        --num_classes 2 \
        --rotate_range 40 \
        --translate_range 0.2  \
        --scale_range 0.8 1.2 \
        --aug_prob 0.5 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.6


############# Experiment 15 #############
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
        --classification_type both_side \
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