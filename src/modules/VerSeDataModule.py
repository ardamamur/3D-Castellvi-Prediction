import pytorch_lightning as pl
import pandas as pd
from utils._prepare_data import DataHandler
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader
import torch


class VerSeDataModule(pl.LightningDataModule):

    def __init__(self, opt, processor:DataHandler, master_list:str):
        super().__init__()
        self.opt = opt
        self.processor = processor
        self.batch_size = opt.batch_size
        self.master_df = pd.read_excel(master_list)
        self.train_records = []
        self.val_records = []
        self.test_records = []
        self.under_sample = opt.under_sample

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        records = self.processor.verse_records
        for record in records:
            if record["split"] == "train":
                self.train_records.append(record)
            elif record["split"] == "val":
                self.val_records.append(record)
            elif record["split"] == "test":
                self.test_records.append(record)
            else:
                raise ValueError("Invalid split value in record: {}".format(record["split"]))
            
        if stage in {'fit', None}:
            self.train_dataset = VerSe(self.opt, self.processor, self.train_records, training=True)
            self.val_dataset = VerSe(self.opt, self.processor, self.val_records, training=False)
        
        if stage in {'test', None}:
            self.train_dataset = VerSe(self.opt, self.processor, self.test_records, training=False)

    def _get_sampler(self, dataset):
        # Compute the true labels for each sample
        true_labels = []
        for index in range(len(dataset)):
            true_labels.append(dataset.__getitem__(index)["class"])

        true_labels = torch.tensor(true_labels)

        # Count the occurrences of each true label
        label_counts = torch.bincount(true_labels)

        # Compute the inverse of the label counts to get the weights
        weights = 1.0 / label_counts.float()

        # Normalize the weights to sum up to the number of classes
        weights = weights / weights.sum() * len(label_counts)

        weights = weights.tolist()

        sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset))
        return sampler


    def train_dataloader(self):

        dataset = VerSe(self.processor, self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True)
        train_dataset, _ = random_split(dataset, self.train_val_split, generator = torch.Generator().manual_seed(42))

        sampler = self._get_sampler(train_dataset)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, sampler = sampler)

    def val_dataloader(self):
        #TODO: Does weighted sampling make sense for validation too?
        dataset = VerSe(self.processor, self.castellvi_classes, self.pad_size, self.use_seg, self.use_binary_classes, training=True)
        _ , val_dataset = random_split(dataset, self.train_val_split, generator = torch.Generator().manual_seed(42))

        
        return DataLoader(val_dataset, batch_size=self.batch_size)
"""
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


