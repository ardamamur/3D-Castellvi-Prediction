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
    def __init__(self, processor:DataHandler, castellvi_classes:list, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, training=True) -> None:
        """
        Initialize an object of 
        """
        # TODO : add new argument for training and testing subject names
        self.processor = processor
        self.pad_size = pad_size
        self.use_seg = use_seg
        self.training = training
        self.binary = use_binary_classes
        self.categories = castellvi_classes
        self.castellvi_dict = {category: i for i, category in enumerate(self.categories)}
        self.transformations = self.get_transformations()
        self.test_transformations = self.get_test_transformations()


    def __len__(self):
        return len(self.processor.verse_records)*2

    def __getitem__(self, index):
        record_index = index // 2  # Each record contributes two data points: original and flipped
        flip = index % 2  # We'll use this to determine if we should flip the image

        record = self.processor.verse_records[record_index]
        print('flip:', flip)
        print('record:', record)
        img = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size)
        img = img[np.newaxis, ...]

        if flip:
            img = np.flip(img, axis=2)  # Assuming flipping is done horizontally

        if self.binary:
            labels = self._get_binary_label(record)
        else:
            labels = self._get_castellvi_right_side_label(record, flip=flip)

        if self.training:
            inputs = self.transformations(img) 
        else:
            inputs = self.test_transformations(img)

        print('label:', labels)
        return {"target": inputs, "class": labels}


    def _get_binary_label(self, record):

        if str(record["castellvi"]) != '0':
            return 1
        else:
            return 0
    
    def _get_castellvi_label(self, record):

        castellvi = str(record["castellvi"])
        one_hot = np.zeros(len(self.categories))    
        one_hot[self.castellvi_dict[castellvi]] = 1
        return one_hot


    def _get_castellvi_right_side_label(self, record, flip):

        castellvi = str(record["castellvi"])
        no_anomaly = ['0', '1a', '1b']
        side = str(record['side'])
        if flip:
            if side is not None:
                if side=='R':
                    side = 'L'
                else:
                    side = 'R'
        # Cause of labels out of range error we mapped the class 2 -> 1 and 3 -> 2
        if castellvi in no_anomaly:
            return 0
        else:
            if castellvi=='2b':
                return 1
            elif castellvi=='3b':
                return 2

            elif castellvi=='2a':
                if side == 'R':
                    return 1
                else:
                    return 0
            # 3B
            else:
                if side=='R':
                    return 2
                else:
                    return 0

    def _get_castellvi_side_labels(self, record):
        castellvi = str(record["castellvi"])


        # side = str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Side'].values[0])
        
        # Split the string into class and subclass (if it exists)
        castellvi_class, castellvi_subclass = None, None
        if len(castellvi) > 1:
            # if the class is not 0 or 4 
            castellvi_class, castellvi_subclass = castellvi[0], castellvi[1]
            if castellvi_subclass.upper() == "A":
               castellvi_subclass = 0
            else:
                castellvi_subclass = castellvi_class
        else:
            # if the class is 0 or 4
            
            # if castellvi_class == '4':
            #     castellvi_class = 2
            #     castellvi_subclass = 3

            castellvi_class = castellvi
            castellvi_subclass = 0

        return [int(castellvi_class), int(castellvi_subclass)]


    def get_transformations(self):

        transformations = montransforms.Compose([montransforms.CenterSpatialCrop(roi_size=[128,86,136]),
                                                montransforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        return transformations
    

    def get_test_transformations(self):
        transformations = montransforms.Compose([montransforms.CenterSpatialCrop(roi_size=[128,86,136]),
                                                montransforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        return transformations
    

        

