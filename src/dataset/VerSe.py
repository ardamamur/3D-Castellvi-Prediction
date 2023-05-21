import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import random
import torchvision
import torchio as tio
from monai import transforms as montransforms
from torch.utils.data import Dataset
from typing import Tuple, Union
from utils._prepare_data import DataHandler
from monai.transforms import Compose, Rotate, Flip, Pad


class VerSe(Dataset):
    def __init__(self, processor:DataHandler, subjects, castellvi_classes:list, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, training=True) -> None:
        """
        Initialize an object of 
        """
        # TODO : add new argument for training and testing subject names
        self.processor = processor
        self.pad_size = pad_size
        self.use_seg = use_seg
        self.training = training
        self.binary = use_binary_classes
        self.bids_subjects = subjects[0]
        self.master_subjects = subjects[1]
        self.categories = castellvi_classes
        self.castellvi_dict = {category: i for i, category in enumerate(self.categories)}
        self.transformations = self.get_transformations()
        self.test_transformations = self.get_test_transformations()


    def __len__(self):
        return len(self.master_subjects)

    def __getitem__(self, index):

        # TODO : Only use multiple families subjects which included in master list
        bids_family = self.bids_subjects[index]
        master_idx = self.master_subjects[index]
        img = self.processor._get_cutout(family=bids_family, return_seg=self.use_seg, max_shape=self.pad_size)
        img = img[np.newaxis, ...]

        if self.binary:
            labels = self._get_binary_label(master_idx)
        else:
            labels = self._get_castellvi_label(master_idx)

        if self.training:
            inputs = self.transformations(img) 
        else:
            inputs = self.test_transformations(img)

        return {"target": inputs, "class": labels}


    def _get_binary_label(self, subject):

        binary_classes = []
        if str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0]) != '0':
            return 1
        else:
            return 0
    
    def _get_castellvi_label(self, subject):

        castellvi = str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0])
        one_hot = np.zeros(len(self.categories))    
        one_hot[self.category_dict[castellvi]] = 1
        return one_hot

    def get_transformations(self):

        transformations = montransforms.Compose([montransforms.transforms.CenterSpatialCrop(128,86,136),
                                                montransforms.transforms.HorizontalFlip(prob=0.5),
                                                montransforms.transforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        return transformations
    

    def get_test_transformations(self):
        transformations = montransforms.Compose([montransforms.transforms.CenterSpatialCrop(128,86,136),
                                                montransforms.transforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        return transformations
    

        

