<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src=https://boneandspine.com/wp-content/uploads/2018/05/lstv.jpg alt="Project logo"></a>
</p>

<h3 align="center">3D Castellvi Prediction</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---


## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [TODOs](#todo)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

The VerSe dataset consists of full-body CT scans enriched with aberrant cases. Anomalies in the lumbosacral region (a rather small part of the whole scan) can be rated via the Castellvi system (see https://radiopaedia.org/articles/castellvi-classification-of-lumbosacral-transitional-vertebrae). Although relatively easy to manually detect, it would still take a lot of time to rate huge datasets by hand. With a given expert grading of the data, the goal is to automate this process and predict the castellvi anomalies with a solid uncertainty estimation.


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

What things you need to install the software and how to install them.

```
~/3D-Castellvi-Prediction$ pip3 install -r requirements.txt
```

### Installing

A step by step series of examples that tell you how to get a development env running.

First create an experiments folder in working directory. 
```
~/3D-Castellvi-Prediction$ mkdir experiments
~/3D-Castellvi-Prediction/experiments$ mkdir baseline_models
~/3D-Castellvi-Prediction/experiments/baseline_models$ mkdir densenet
~/3D-Castellvi-Prediction/experiments/baseline_models/densenet$ mkdir best_models
~/3D-Castellvi-Prediction/experiments/baseline_models/densenet$ mkdir lightning_logs

```

Update settings based on your local environment by using the following file. 
```
~/3D-Castellvi-Prediction/src/utils/environment_settings.py
```


Install BIDS toolbox
* https://bids-standard.github.io/bids-starter-kit/index.html

After installed the BIDS, add it to your system path. You can run following command for it. 
```
export PYTHONPATH=$PYTHONPATH:<path_to_bids>/bids
```

## Running the training <a name = "tests"></a>

```
Training settings

  --data_root  Path to the data root
  --data_types Data types to use (rawdata, derivatives)
  --img_types  Image types to use (ct, subreg, cortex)
  --master_list Path to the master list
  --classification_type Classification type (right_side, right_side_binary, both_side)
  --castellvi_classes CASTELLVI_CLASSES

  --model MODEL(densenet and/or resnet)
  --scheduler (ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts)
  --optimizer Optimizer to use (AdamW, SGD)
  --learning_rate
  --weight_decay
  --total_iterations
  --batch_size
  --accumulate_grad_batches
  --num_workers
  --save_intervals
  --n_epochs
  --experiments Path to the experiments folder
  --num_classes NUM_CLASSES ( 2 and or 3 )
  --val_metric Validation metric to use in monmitoring (val_loss, val_acc, val_mcc)

  --rotate_range 
  --shear_range
  --translate_range 
  --scale_range
  --aug_prob 
  --elastic_transform   Use Rand3DElastic for augmentations
  --sigma_range 
  --magnitude_range MAGNITUDE_RANGE 


  --use_seg             Use segmentation
  --use_bin_seg         Use binary segmentation
  --use_seg_and_raw     Use segmentation and raw data
  --no_cuda             Do not use cuda
  --weighted_sample     Use weighted sampling
  --weighted_loss       Use weighted loss
  --flip_all            Flip all images
  --cross_validation    Cross validation
  --use_zero_out        Use zero out
  --gradual_freezing    Gradual freezing
  --dropout_prob        Dropout probability
```

* You can run the following command based on your settings or you can update the bash script (run_experiments.sh) to run multiple experiments at the same time. 

```

python3 train.py \
        --flip_all \
        --use_seg \
        --use_bin_seg \
        --use_zero_out \
        --weighted_sample \
        --n_epochs 85 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --weight_decay 0.001 \
        --classification_type right_side \
        --model densenet \
        --num_classes 3 \
        --rotate_range 40 \
        --translate_range 0.6  \
        --scale_range 0.9 1.1 \
        --aug_prob 1 \
        --num_workers 8 \
        --elastic_transform \
        --val_metric val_mcc \
        --dropout_prob 0.2
```


## Running the evaluation <a name = "tests"></a>
You can evaluate your model with the following command 
```
python3 eval.py \
        --model_path <path_to_your_model_checkpoint>
        --version <experiment_number> \

```
Results will be saved in a csv file with the following experiment no. 


# Authors
* [Melisa Ankut](https://github.com/melisaankut) 
* [Arda Mamur](https://github.com/ardamamur)
* [Daniel Regenbrecht](https://github.com/doppelplusungut)
