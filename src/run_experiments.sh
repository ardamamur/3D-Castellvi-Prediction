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


## Hyperparameter optimizationg for Data Augmentation
python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 10 --shear_range 0.2 --translate_range 0.15 --scale_range 0.9 1.1
python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 17

python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 15 --shear_range 0.2 --translate_range 0.20 --scale_range 0.9 1.1
python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 18

python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 20 --shear_range 0.2 --translate_range 0.25 --scale_range 0.9 1.1
python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 19

python3 train.py --use_seg --flip_all --classification_type right_side --model densenet --weighted_sample --num_classes 3 --rotate_range 5 --shear_range 0.2 --translate_range 0.10 --scale_range 0.9 1.1
python3 eval_V2.py --use_seg --classification_type right_side --model densenet --num_classes 3 --weighted_sample --version_no 20