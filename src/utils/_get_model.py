import torch
from torch import nn
from models.ResNet3D import *

def _get_num_classes(binary_classification, castellvi_classes):
    if binary_classification:
        num_classes = 1
    else:
        num_classes = len(castellvi_classes)
    return num_classes

def _generate_model(model:str, num_classes:int, no_cuda:bool=False):
    if model == 'resnet':
        model = get_resnet_model(num_classes=num_classes, no_cuda=no_cuda)
    else:
        raise Exception('Not Implemented')
    
    return model
