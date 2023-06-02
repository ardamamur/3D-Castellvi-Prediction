#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"


# Experiments :
#     0. DenseNet	- Multi Class - Right side
#     1. DenseNet	- Multi Class - Right side, weighted sample
#     2. DenseNet	- Multi Class - Right side, weighted loss
#     3. DenseNet	- Multi Class - Right side, over_sampling
#     4. DenseNet	- Multi Class - Right side, weighted sample, over_sampling
#     5. DenseNet	- Multi Class - Right side, weighted loss, over_sampling 



####################### EXPERIMENTS - FLIPPING ONLY  A-B CASES  ############################

# Experiment 0
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx \
--classification_type right_side \
--model densenet \
--weighted_sample False \
--weighted_loss False \
--num_classes 3


# Experiment 1
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx \
--classification_type right_side \
--model densenet \
--weighted_sample True \
--weighted_loss False \
--num_classes 3


# Experiment 2
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx \
--classification_type right_side \
--model densenet \
--weighted_sample False \
--weighted_loss True \
--num_classes 3

####################### EXPERIMENTS - FLIPPING ALL IMAGES ############################

# Experiment 3
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx \
--classification_type right_side \
--model densenet \
--weighted_sample False \
--weighted_loss False \
--num_classes 3

# Experiment 4
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx \
--classification_type binary \
--model densenet \
--weighted_sample True \
--weighted_loss False \
--num_classes 3

# Experiment 5
python3 train.py --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx \
--classification_type binary \
--model densenet \
--weighted_sample False \
--weighted_loss True \
--num_classes 3