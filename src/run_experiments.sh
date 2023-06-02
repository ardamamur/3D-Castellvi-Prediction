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



############################################    FLIP ONLY A-B CASES    #########################################################################
# Experiment 0
python3 train.py --flip_all False --classification_type right_side --model densenet --weighted_sample False --weighted_loss False --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 0
#################################################################################################################################################
# Experiment 1
python3 train.py --flip_all False --classification_type right_side --model densenet --weighted_sample True --weighted_loss False --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 1
#################################################################################################################################################
# Experiment 2
python3 train.py --flip_all False --classification_type right_side --model densenet --weighted_sample False --weighted_loss True --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 2
#################################################################################################################################################


#################################################   FLIP ALL IMAGES    #########################################################################
# Experiment 3
python3 train.py --flip_all True --classification_type right_side --model densenet --weighted_sample False --weighted_loss False --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 3
#################################################################################################################################################
# Experiment 4
python3 train.py --flip_all True --classification_type right_side --model densenet --weighted_sample True --weighted_loss False --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 4
#################################################################################################################################################
# Experiment 5
python3 train.py --flip_all True --classification_type right_side --model densenet --weighted_sample False --weighted_loss True --num_classes 3
python3 eval_V2.py --classification_type right_side --model densenet --num_classes 3 --version_no 5