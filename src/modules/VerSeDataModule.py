from typing import Dict, Sequence
import pytorch_lightning as pl
import torch
import pandas as pd
#from monai.metrics import MSEMetric, MAEMetric, RMSEMetric, compute_auc_roc
from utils._prepare_data import DataHandler, save_list
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class VerSeDataModule(pl.LightningDataModule):

    def __init__(self, processor:DataHandler, subjects, castellvi_classes:list, master_list:str, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, batch_size=32, train_val_test_split=[0.8, 0.1, 0.1]):
        super().__init__()
        self.processor = processor
        self.subjects = subjects
        self.castellvi_classes = castellvi_classes # It makes sense to merge 3 and 4. Because 4 includes 3 inside. 
        self.pad_size = pad_size
        self.use_seg = use_seg
        self.use_binary_classes = use_binary_classes
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.master_df = pd.read_excel(master_list)



    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Split your dataset here
        bids_subjects, master_subjects = self.subjects
        bids_train, bids_val, bids_test = [], [], []
        master_train, master_val, master_test = [], [], []

        for i in range(len(master_subjects)):
            if self.master_df.loc[self.master_df['Full_Id'] == master_subjects[i]]['Split'].values == 'train':
                master_train.append(master_subjects[i])
                bids_train.append(bids_subjects[i])
            elif self.master_df.loc[self.master_df['Full_Id'] == master_subjects[i]]['Split'].values == 'val':
                master_val.append(master_subjects[i])
                bids_val.append(bids_subjects[i])
            else:
                master_test.append(master_subjects[i])
                bids_test.append(bids_subjects[i])            

        if stage in {'fit', None}:
            self.train_dataset = VerSe(self.processor, (bids_train, master_train), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True)
            self.val_dataset = VerSe(self.processor, (bids_val, master_val), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=False)
            
        if stage in {'test', None}: 
            self.test_dataset = VerSe(self.processor, (bids_test, master_test), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
