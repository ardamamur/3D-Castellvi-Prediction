from typing import Dict, Sequence
import pytorch_lightning as pl
import torch
#from monai.metrics import MSEMetric, MAEMetric, RMSEMetric, compute_auc_roc
from utils._prepare_data import DataHandler
from dataset import VerSe
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split


class VerSeDataModule(pl.LightningDataModule):

    def __init__(self, processor:DataHandler, subjects, castellvi_classes:list, pad_size=(128,86,136), use_seg=False, use_binary_classes=True, batch_size=32):
        super().__init__()
        self.processor = processor
        self.subjects = subjects
        self.castellvi_classes = castellvi_classes # It makes sense to merge 3 and 4. Because 4 includes 3 inside. 
        self.pad_size = pad_size
        self.use_seg = use_seg
        self.use_binary_classes = use_binary_classes
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, phase=None):
        # Split your dataset here
        bids_subjects, master_subjects = self.subjects
        bids_train_val, bids_test, master_train_val, master_test = train_test_split(bids_subjects, master_subjects, test_size=self.train_val_test_split[2], random_state=42)
        bids_train, bids_val, master_train, master_val = train_test_split(bids_train_val, master_train_val, test_size=self.train_val_test_split[1]/(self.train_val_test_split[0]+self.train_val_test_split[1]), random_state=42)

        if phase == 'train':
            self.train_dataset = VerSe(self.processor, (bids_train, master_train), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True, apply_transform=True)
            self.val_dataset = VerSe(self.processor, (bids_val, master_val), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True, apply_transform=False)
            
        if phase == 'test':
            self.test_dataset = VerSe(self.processor, (bids_test, master_test), self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
