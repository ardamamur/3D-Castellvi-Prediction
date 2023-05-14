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
from _get_data import DataHandler


class VerSe(Dataset):
    def __init__(self, crop_size, processor:DataHandler, subjects, use_seg=True, use_binary_classes=True, training=True) -> None:
        """
        Initialize an object of 
        """
        self.processor = processor
        self.crop_size = crop_size
        self.use_seg = use_seg
        self.training = training
        self.binary = use_binary_classes
        self.bids_subjects = subjects[0]
        self.master_subjects = subjects[1]
        self.transform = self.get_transformations()
    
    def __len__(self):
        return len(self.master_subjects)

    def __getitem__(self, index):

        bids_idx = self.bids_subjects[index]
        master_idx = self.master_subjects[index]
        family = self.processor._get_subject_family(subject=bids_idx)
        last_l = self.processor.master_df.loc[self.processor.master_df['Full_Id'] == master_idx, 'Last_L'].values
        roi_object_idx = self.processor._get_roi_object_idx(roi_parts=[last_l, 'S1'])
        img = self.processor._get_cutout(family=family, roi_object_idx=roi_object_idx, return_seg=self.use_seg)
        if self.binary:
            labels = self._get_binary_labels()
        else:
            labels = self._get_castellvi_labels()
        
        if self.training:
            img = self.transform(img)

        return img, labels
    

    def _get_binary_labels(self):
        binary_classes = []

        # TODO change it based on the order of master_subjects

        for i in self.processor.master_df['Castellvi'].values.tolist():
            if str(i) is not '0':
                binary_classes.append(1)
            else:
                binary_classes.append(2)
        return np.array(binary_classes)
    
    def _get_castellvi_labels(self):

        # Use get_dummies to create one-hot encoding
        #df_encoded = pd.get_dummies(df['Castellvi'])
        #y_values = df_encoded.values
        #return y_values
        pass

    
    def get_transformations(self):
        transformations = tio.Compose([montransforms.RandSpatialCrop(roi_size=self.crop_size, random_center=True, random_size=False),
                                       tio.RandomFlip(axes=(0, 1, 2)),
                                       tio.RandomAffine(scales=0.1, isotropic=True)
                                       ])
        return transformations
    

    def get_test_transformations(self):
        return montransforms.SpatialPad((-1, -1, 160))

        

