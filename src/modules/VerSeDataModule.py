from typing import Dict, Sequence
import pytorch_lightning as pl
import torch
import pandas as pd
#from monai.metrics import MSEMetric, MAEMetric, RMSEMetric, compute_auc_roc
from utils._prepare_data import DataHandler, save_list
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split


class VerSeDataModule(pl.LightningDataModule):

    def __init__(self, processor:DataHandler, castellvi_classes:list, master_list:str, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, batch_size=32, train_val_split=[0.8, 0.2]):
        super().__init__()
        self.processor = processor
        self.castellvi_classes = castellvi_classes # It makes sense to merge 3 and 4. Because 4 includes 3 inside. 
        self.pad_size = pad_size
        self.use_seg = use_seg
        self.use_binary_classes = use_binary_classes
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.master_df = pd.read_excel(master_list)



    def prepare_data(self):
        pass

    def setup(self, stage=None):
        #TODO: Read a saved train/val/test split
        pass

    def train_dataloader(self):
        dataset = VerSe(self.processor, self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True)
        train_dataset, _ = random_split(dataset, self.train_val_split, generator = torch.Generator().manual_seed(42))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = VerSe(self.processor, self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True)
        _ , val_dataset = random_split(dataset, self.train_val_split, generator = torch.Generator().manual_seed(42))
        
        return DataLoader(val_dataset, batch_size=self.batch_size)
"""
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
"""