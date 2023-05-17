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


class VerSe(Dataset):
    def __init__(self, processor:DataHandler, subjects, castellvi_classes:list, pad_size=(128,128,128), use_seg=False, use_binary_classes=True, training=True, apply_transform=True) -> None:
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
        self.apply_transform = apply_transform
        self.transform = self.get_transformations
        self.test_transform = self.get_test_transformations

    def __len__(self):
        return len(self.master_subjects)

    def __getitem__(self, index):
        # TODO : Do not apply cutout extraction for test images
        bids_idx = self.bids_subjects[index]
        master_idx = self.master_subjects[index]
        family = self.processor._get_subject_family(subject=bids_idx)
        last_l = self.processor.master_df.loc[self.processor.master_df['Full_Id'] == master_idx, 'Last_L'].values
        roi_object_idx = self.processor._get_roi_object_idx(roi_parts=[last_l, 'S1'])
        img = self.processor._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=self.use_seg, pad=True, pad_size=self.crop_size)
        if self.binary:
            labels = self._get_binary_label(master_idx)
        else:
            labels = self._get_castellvi_label(master_idx)
        
        if self.apply_transform:
            img = self.transform(img)
        else:
            img = self.test_transform(img)

        return img, labels
    

    def _get_binary_label(self, subject):

        binary_classes = []
        if str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0]) is not '0':
            binary_classes.append(1)
        else:
            binary_classes.append(0)
        return np.array(binary_classes) 
    
    def _get_castellvi_label(self, subject):

        castellvi = str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0])
        one_hot = np.zeros(len(self.categories))    
        one_hot[self.category_dict[castellvi]] = 1
        return one_hot


    
    def get_transformations(self):
        # TODO : Ask if it makes sense to apply random cropping to cutout ? 
        transformations = tio.Compose([montransforms.RandSpatialCrop(roi_size=self.pad_size, random_center=True, random_size=False),
                                       tio.RandomFlip(axes=(0, 1, 2)),
                                       tio.RandomAffine(scales=0.1, isotropic=True)
                                       ])
        return transformations
    

    def get_test_transformations(self):
        # TODO : Ask if it makes sense to apply spatialpad to test data
        return montransforms.SpatialPad((-1, -1, 160))

        

