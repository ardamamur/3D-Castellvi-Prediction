#!/bin/bash
# train_model.sh

# set the environment variable
export PYTHONWARNINGS="ignore"
export PYTHONPATH=$PYTHONPATH:/data1/practical-sose23/castellvi/castellvi_prediction/bids


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_0/densenet-epoch=38-val_mcc=0.91.ckpt \
        --version 0 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_1/densenet-epoch=83-val_mcc=0.90.ckpt \
        --version 1 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_2/densenet-epoch=62-val_mcc=0.84.ckpt \
        --version 2 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_3/densenet-epoch=25-val_mcc=0.87.ckpt \
        --version 3 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_4/densenet-epoch=30-val_mcc=0.91.ckpt \
        --version 4 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_5/densenet-epoch=37-val_mcc=0.93.ckpt \
        --version 5 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_6/densenet-epoch=65-val_mcc=0.89.ckpt \
        --version 6 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_7/densenet-epoch=65-val_mcc=0.86.ckpt \
        --version 7 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_8/densenet-epoch=81-val_mcc=0.89.ckpt \
        --version 8 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_9/densenet-epoch=41-val_mcc=0.89.ckpt \
        --version 9 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_10/densenet-epoch=09-val_mcc=0.89.ckpt \
        --version 10 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_11/densenet-epoch=131-val_mcc=0.85.ckpt \
        --version 11 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_12/densenet-epoch=78-val_mcc=0.90.ckpt \
        --version 12 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_13/densenet-epoch=157-val_mcc=0.70.ckpt \
        --version 13 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_14/densenet-epoch=96-val_mcc=0.86.ckpt \
        --version 14 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_15/densenet-epoch=69-val_mcc=0.83.ckpt \
        --version 15 \


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_16/densenet-epoch=82-val_mcc=0.83.ckpt \
        --version 16 \


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_17/densenet-epoch=75-val_mcc=0.87.ckpt \
        --version 17 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_18/densenet-epoch=03-val_mcc=0.92.ckpt \
        --version 18 \


#####################################



python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_0/densenet-epoch=38-val_mcc=0.91.ckpt \
        --split test \
        --version 0 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_1/densenet-epoch=83-val_mcc=0.90.ckpt \
        --split test \
        --version 1 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_2/densenet-epoch=62-val_mcc=0.84.ckpt \
        --split test \
        --version 2 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_3/densenet-epoch=25-val_mcc=0.87.ckpt \
        --split test \
        --version 3 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_4/densenet-epoch=30-val_mcc=0.91.ckpt \
        --split test \
        --version 4 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_5/densenet-epoch=37-val_mcc=0.93.ckpt \
        --split test \
        --version 5 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_6/densenet-epoch=65-val_mcc=0.89.ckpt \
        --split test \
        --version 6 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_7/densenet-epoch=65-val_mcc=0.86.ckpt \
        --split test \
        --version 7 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_8/densenet-epoch=81-val_mcc=0.89.ckpt \
        --split test \
        --version 8 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_9/densenet-epoch=41-val_mcc=0.89.ckpt \
        --split test \
        --version 9 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_10/densenet-epoch=09-val_mcc=0.89.ckpt \
        --split test \
        --version 10 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_11/densenet-epoch=131-val_mcc=0.85.ckpt \
        --split test \
        --version 11 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_12/densenet-epoch=78-val_mcc=0.90.ckpt \
        --split test \
        --version 12 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_13/densenet-epoch=157-val_mcc=0.70.ckpt \
        --split test \
        --version 13 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_14/densenet-epoch=96-val_mcc=0.86.ckpt \
        --split test \
        --version 14 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_15/densenet-epoch=69-val_mcc=0.83.ckpt \
        --split test \
        --version 15 \


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_16/densenet-epoch=82-val_mcc=0.83.ckpt \
        --split test \
        --version 16 \


python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_17/densenet-epoch=75-val_mcc=0.87.ckpt \
        --split test \
        --version 17 \

python3 eval.py \
        --model_path /data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/best_models/version_18/densenet-epoch=03-val_mcc=0.92.ckpt \
        --split test \
        --version 18 \