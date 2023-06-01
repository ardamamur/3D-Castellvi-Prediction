#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"


""" 
Experiments :
    1. DenseNet	- Binary - Both side
    2. DenseNet	- Multi Class - Right side
    3. DenseNet	- Multi Class - Right side, weighted loss
    4. DenseNet	- Multi Class - Right side, weighted sample
    5. DenseNet	- Multi Class - Right side, over_sampling
    6. DenseNet	- Multi Class - Right side, weighted loss, over_sampling
    7. DenseNet	- Multi Class - Right side, weighted sample, over_sampling 

"""


############################################# EXPERIMENT 1 ######################################################################
# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="binary"

param3="weighted_loss"
new_value3="False"

param4="weighted_sample"
new_value4="False"


# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml



############################################# EXPERIMENT 2 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="False"

param4="weighted_sample"
new_value4="False"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

############################################# EXPERIMENT 3 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="True"

param4="weighted_sample"
new_value4="False"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

############################################# EXPERIMENT 4 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="False"

param4="weighted_sample"
new_value4="True"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V3.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

###################################################################################################################################



############################################# EXPERIMENT 5 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="False"

param4="weighted_sample"
new_value4="False"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

############################################# EXPERIMENT 6 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="True"

param4="weighted_sample"
new_value4="False"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

############################################# EXPERIMENT 7 ######################################################################

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="False"

param4="weighted_sample"
new_value4="True"

param5="master_list"
new_value5="/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param4:\s*\).*$/\1$new_value4/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param5:\s*\).*$/\1$new_value5/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

###################################################################################################################################


# Evaluate
# python3 eval_V2.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
