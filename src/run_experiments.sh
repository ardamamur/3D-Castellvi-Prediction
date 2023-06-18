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

# ############################################    FLIP ONLY A-B CASES    #########################################################################
# # Experiment 0
# python3 train.py --use_seg --classification_type right_side --model densenet --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --version_no 6
# #################################################################################################################################################
# # Experiment 1
# python3 train.py --use_seg --classification_type right_side --model densenet --weighted_sample --num_classes 3
# python3 eval_V2.py --use_seg  --classification_type right_side --model densenet --num_classes 3 --version_no 7
# #################################################################################################################################################
# # Experiment 2
# python3 train.py --use_seg --classification_type right_side --model densenet --weighted_loss --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 8
# #################################################################################################################################################


# #################################################   FLIP ALL IMAGES    #########################################################################
# # Experiment 3
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --version_no 9
# #################################################################################################################################################
# # Experiment 4
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --version_no 10
# #################################################################################################################################################
# # Experiment 5
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_loss --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 11




# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 0
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 1
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 2
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 3
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 4
# python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 5


# ######################

# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --version_no 6
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 7
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 8
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --version_no 9
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 10
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_loss --version_no 11


## Cross validation
#python3 train.py --use_seg --flip_all --cross_validation --classification_type right_side --model densenet --weighted_sample --num_classes 3

# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 12
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 13
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 14
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 15
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 16

## Hyperparameter optimizationg for Data Augmentation
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 10 --shear_range 0.2 --translate_range 0.15 --scale_range 0.9 1.1
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 17

# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 15 --shear_range 0.2 --translate_range 0.20 --scale_range 0.9 1.1
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 18

# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --shear_range 0.2 --translate_range 0.25 --scale_range 0.9 1.1
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 19

# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 5 --shear_range 0.2 --translate_range 0.10 --scale_range 0.9 1.1
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 20


## Test Special Cases
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --master_list /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V5.xlsx
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 21


# ResNet
# #python3 train.py --use_seg --flip_all --classification_type right_side --model resnet --weighted_sample --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model resnet --num_classes 3 --version_no 0

# #python3 train.py --use_seg --flip_all --classification_type right_side --model pretrained_resnet --weighted_sample --num_classes 3
# python3 eval_V2.py --use_seg --classification_type right_side --model pretrained_resnet --num_classes 3 --version_no 0




# ############################ HPO FOR DATA AUGMENTATION ##########################################
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.2 # version_22
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 40 --translate_range 0.2 # version_23
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2 # version_24
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.2 # version_25
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.4 # version_26
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.6 # version_27

# ############################ HPO FOR DATA ZEROING_OUT ##########################################

# ########## SEG IMAGES #########
# python3 train.py --use_seg --use_bin_seg --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.2 # version_28
# python3 train.py --use_seg --use_bin_seg --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 40 --translate_range 0.2 # version_29
# python3 train.py --use_seg --use_bin_seg --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2 # version_30

# ########## NON-SEG IMAGES ######
# python3 train.py --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --translate_range 0.2 # version_31
# python3 train.py --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 40 --translate_range 0.2 # version_32
# python3 train.py --use_zero_out --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2 # version_33





############################### EXPERIMENTS 17.06.2023 ##########################################

########## DenseNet - Zeroing Parts - Weighted Sampling ############
# python3 train.py --use_seg --use_bin_seg --use_zero_out  --weighted_sample --batch_size 8 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_35
# python3 train.py --use_seg --use_bin_seg --use_zero_out  --weighted_sample --batch_size 32 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_36
# python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out  --weighted_sample --batch_size 8 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_37
#python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out  --weighted_sample --batch_size 32 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_38


# ######### ResNet ############
# python3 train.py --flip_all --use_seg --weighted_sample --batch_size 8 --n_epochs 200 --classification_type right_side --model pretrained_resnet --model_type resnet18 --unfreeze_top --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 1
# python3 train.py --flip_all --use_seg --weighted_sample --batch_size 8 --n_epochs 200 --classification_type right_side --model pretrained_resnet --model_type resnet50 --unfreeze_top --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 2
# python3 train.py --flip_all --use_seg --weighted_sample --batch_size 8 --n_epochs 200 --classification_type right_side --model pretrained_resnet --model_type resnet101 --unfreeze_top --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 3
# python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out  --weighted_sample --batch_size 8 --n_epochs 200 --classification_type right_side --model pretrained_resnet --model_type resnet18 --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 4


# ######### DenseNet with Rand3DElastic
# python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --elastic_transform --rotate_range 60 --shear_range 0.2 --translate_range 0.20 --scale_range 0.9 1.1 # version 39
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 39

# python3 train.py --use_seg --use_bin_seg --use_zero_out --classification_type right_side --model densenet --weighted_sample --num_classes 3 --elastic_transform --rotate_range 60 --shear_range 0.2 --translate_range 0.20 --scale_range 0.9 1.1 # version 40
# python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 40



 # python3 train.py --flip_all --use_seg --weighted_sample --batch_size 8 --n_epochs 250 --classification_type right_side --model pretrained_resnet --model_type resnet18  --gradual_freezing --weighted_sample --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 6



############### EXPERIMENTS 18.06.2021 #####################
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 20 --translate_range 0.2  --aug_prob 0.5 # version_0
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 40 --translate_range 0.2  --aug_prob 0.5 # version_1
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_2
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 20 --translate_range 0.2  --aug_prob 0.5 # version_3
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 20 --translate_range 0.4  --aug_prob 0.5 # version_4
python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 20 --translate_range 0.6  --aug_prob 0.5 # version_5

python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_6
python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_7
python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_8

python3 train.py --flip_all --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 20 --translate_range 0.2  --aug_prob 0.5 # version_9
python3 train.py --flip_all --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 40 --translate_range 0.2  --aug_prob 0.5 # version_10
python3 train.py --flip_all --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_11

python3 train.py --flip_all --use_seg --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --elastic_transform --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_12
python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --elastic_transform --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_13
python3 train.py --flip_all --use_zero_out --weighted_sample --n_epochs 70 --batch_size 16 --learning_rate 0.001 --classification_type right_side --model densenet --num_classes 3 --elastic_transform --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 # version_14


####### ResNet ########
python3 train.py --flip_all --use_seg --weighted_sample --batch_size 16 --n_epochs 120 --learning_rate 0.0001 --classification_type right_side --model pretrained_resnet --model_type resnet18  --gradual_freezing --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 6
python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --batch_size 16 --n_epochs 120 --learning_rate 0.0001 --classification_type right_side --model pretrained_resnet --model_type resnet18  --gradual_freezing --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 6
python3 train.py --flip_all --use_zero_out --weighted_sample --batch_size 16 --n_epochs 120 --learning_rate 0.0001 --classification_type right_side --model pretrained_resnet --model_type resnet18  --gradual_freezing --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 6

python3 train.py --flip_all --use_seg --use_bin_seg --use_zero_out --weighted_sample --batch_size 16 --n_epochs 120 --learning_rate 0.0001 --classification_type right_side --model resnet --model_type resnet18  --gradual_freezing --num_classes 3 --rotate_range 60 --translate_range 0.2  --aug_prob 0.5 #version 6
