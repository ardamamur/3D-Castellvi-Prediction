#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids
export PYTHONWARNINGS="ignore"

python3 train.py --settings /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml
