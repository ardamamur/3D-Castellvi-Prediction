<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">3D Castellvi Prediction</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Few lines describing your project.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

The VerSe dataset consists of full-body CT scans enriched with aberrant cases. Anomalies in the lumbosacral region (a rather small part of the whole scan) can be rated via the Castellvi system (see https://radiopaedia.org/articles/castellvi-classification-of-lumbosacral-transitional-vertebrae). Although relatively easy to manually detect, it would still take a lot of time to rate huge datasets by hand. With a given expert grading of the data, the goal is to automate this process and predict the castellvi anomalies with a solid uncertainty estimation.


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them.

```
~/3D-Castellvi-Prediction$ pip3 install -r requirements.txt
```

### Installing

A step by step series of examples that tell you how to get a development env running.

First create an experiments folder in working dir
```
~/3D-Castellvi-Prediction$ mkdir experiments
~/3D-Castellvi-Prediction/experiments$ mkdir baseline_models
~/3D-Castellvi-Prediction/experiments/baseline_models$ mkdir densenet
~/3D-Castellvi-Prediction/experiments/baseline_models/densenet$ mkdir best_models
~/3D-Castellvi-Prediction/experiments/baseline_models/densenet$ mkdir lightning_logs

```

Update settings based on your local environment
```
data_root : <abs_path of dataset folders>
master_list : <abs_path of master_list file where you located>
experiments : <abs_path of experiments folder>
test_data_path : <abs path of where you want to save test image names>
```

Update train_model.sh
```
1. change bids toolbox path in python_env section
2. update the path of settings.yaml file 
```


## 🔧 Running the training <a name = "tests"></a>


Run the bashscript

```
~/3D-Castellvi-Prediction/src$ bash train_model.sh
```
