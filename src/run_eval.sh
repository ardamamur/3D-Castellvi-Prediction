#!/bin/bash
# train_model.sh

# set the environment variable
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_0/densenet-epoch=20-val_mcc=0.88.ckpt \
        --version 0 \
