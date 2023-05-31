#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"


# First Experiment

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="True"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml


python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml


# Second Experiment

# Parameters to update and new values
param1="model"
new_value1="densenet"

param2="classification_type"
new_value2="right_side"

param3="weighted_loss"
new_value3="False"

# Update the YAML file
sed -i.bak "s/^\($param1:\s*\).*$/\1$new_value1/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param2:\s*\).*$/\1$new_value2/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
sed -i.bak "s/^\($param3:\s*\).*$/\1$new_value3/" /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml


# Evaluate
python3 eval_V2.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
