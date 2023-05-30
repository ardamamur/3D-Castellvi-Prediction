import torch
from torch import nn
from models.ResNet3D import *
import pandas as pd

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

def _get_weights(master_list, rigth_side):
    if rigth_side:

        no_side = ['0', '1a', '1b', '4']
        side_2 = ['2a', '2b']
        side_3 = ['3a', '3b']

        master_df = pd.read_excel(master_list)
        counts0 = [len(master_df[master_df['Castellvi'].astype(str) == c]) for c in no_side]
        counts2 = [len(master_df[master_df['Castellvi'].astype(str) == c]) for c in side_2]
        counts3 = [len(master_df[master_df['Castellvi'].astype(str) == c]) for c in side_3]
        total_count0 = sum(counts0)
        total_count2 = sum(counts2)
        total_count3 = sum(counts3)

        counts = [total_count0, total_count2, total_count3]
        weights = [1 / c for c in counts]
        return weights
    else:
        raise Exception('Not Implemented')