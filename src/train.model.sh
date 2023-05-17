#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3

# call your training script with all the parameters
python3 train_script.py \
--data_root /data1/practical-sose23/dataset-verse19 /data1/practical-sose23/dataset-verse20 \
--data_types rawdata derivatives \
--img_types ct subreg cortex \
--master_list ../dataset/VerSe_masterlist.xlsx \
--binary_classification True \
--castellvi_classes 1a 1b 2a 2b 3a 3b 4 0 \
--use_seg False \
--phase train \
--learning_rate 0.001 \
--weight_decay 0.0001 \
--total_iterations 100 \
--batch_size 8 \
--n_epochs 200 \
--no_cuda False \
--gpu_id 3 \
--n_devices 1 \
--model resnet \
--manual_seed 1
