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
    def __init__(self, processor:DataHandler, subjects, castellvi_classes:list, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, training=True, apply_transform=True) -> None:
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

        self.transform = tio.Compose([montransforms.RandSpatialCrop(roi_size=self.pad_size, random_center=True, random_size=False),
                                       tio.RandomAffine(degrees=(10, 10, 10), scales=0.1, isotropic=True) # added random rotation
                                    ])


    def __len__(self):
        return len(self.master_subjects)

    def __getitem__(self, index):
        # TODO : Do not apply cutout extraction for test images
        # TODO : Only use multiple families subjects which included in master list

        #family = self.processor._get_subject_family(subject=bids_idx)
        #last_l = self.processor.master_df.loc[self.processor.master_df['Full_Id'] == master_idx, 'Last_L'].values
        #roi_object_idx = self.processor._get_roi_object_idx(roi_parts=[last_l, 'S1'])
        #print("Getting item", index)

        bids_family = self.bids_subjects[index]
        print( self.bids_subjects[index])
        #print(bids_family)

        master_idx = self.master_subjects[index]
        img = self.processor._get_cutout(family=bids_family, return_seg=self.use_seg, max_shape=self.pad_size)
        img = img[np.newaxis, ...]
        #print(img.shape)
        if self.binary:
            labels = self._get_binary_label(master_idx)
        else:
            labels = self._get_castellvi_label(master_idx)


        if self.transform is not None:
            inputs = self.transform(img)  # convert PIL Image to numpy array
        
        else:
            inputs = torch.from_numpy(inputs)  # convert numpy array to PyTorch tensor

        return {"target": inputs, "class": labels}
        #labels = np.array(labels)
        #labels = label.astype(np.float32)


        # if self.apply_transform:
        #     transform = self.get_transformations()
        #     img = transform(img)
        # else:
        #     transform = self.get_test_transformations()
        #     img = transform(img)

        #return img, labels
    

    def _get_binary_label(self, subject):

        binary_classes = []
        if str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0]) != '0':
            #binary_classes.append(1)
            return 1
        else:
            #binary_classes.append(0)
            return 0
        #return np.array(binary_classes) 
    
    def _get_castellvi_label(self, subject):

        castellvi = str(self.processor.master_df.loc[self.processor.master_df['Full_Id'] == subject]['Castellvi'].values[0])
        one_hot = np.zeros(len(self.categories))    
        one_hot[self.category_dict[castellvi]] = 1
        return one_hot


    # Search for random rotation !!! -> see albumentations / kornia --> added random rotation by using torchio.randomaffine
    
    def get_transformations(self):
        # TODO : Ask if it makes sense to apply random cropping to cutout ?
        # ANS : If you have a margin in core area than it make sense to apply transformations 
        """transformations = tio.Compose([montransforms.RandSpatialCrop(roi_size=self.pad_size, random_center=True, random_size=False),
                                       #tio.RandomFlip(axes=(0, 1, 2)), # this does not make any sense (up and down doesnt make sense but left -> right (especcialy in 'a' cases) make sense (but you have to be careful for castellvi classes)). Always be sure to have correct ground truth 
                                       tio.RandomFlip(axes=(2,)),
                                       tio.RandomAffine(degrees=(10, 10, 10), scales=0.1, isotropic=True) # added random rotation
                                       ])"""
        transformations = montransforms.Compose([montransforms.transforms.CenterSpatialCrop(128,86,136),
                                                montransforms.transforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        return transformations
    

    def get_test_transformations(self):
        # TODO : Ask if it makes sense to apply spatialpad to test data
        # center crop 
        transformations = montransforms.Compose([montransforms.transforms.CenterSpatialCrop(128,86,136),
                                                montransforms.transforms.RandRotate(range_x = 0.2, range_y = 0.2, range_z = 0.2, prob = 0.5)
                                                ])
        #transformations = tio.transforms.CropOrPad((128,86,136))  # Adjust the size according to your needs
        return transformations
    

        

