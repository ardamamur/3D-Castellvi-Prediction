#!/bin/bash
# train_model.sh

# set the environment variable
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids

python3 train.py --settings /u/home/mamar/3D-Castellvi-Prediction/settings.yaml
