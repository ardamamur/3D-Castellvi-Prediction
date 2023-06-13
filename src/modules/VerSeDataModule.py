import pytorch_lightning as pl
import pandas as pd
from utils._prepare_data import DataHandler
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader, random_split
import torch


class VerSeDataModule(pl.LightningDataModule):

    def __init__(self, opt, processor:DataHandler):
        super().__init__()
        self.opt = opt
        self.processor = processor
        self.batch_size = opt.batch_size
        self.train_records = []
        self.val_records = []
        self.test_records = []
        self.weighted_sample = opt.weighted_sample
        self.num_workers = opt.num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        records = self.processor.verse_records
        for record in records:
            if record["dataset_split"] == "train":
                self.train_records.append(record)
            elif record["dataset_split"] == "val":
                self.val_records.append(record)
            elif record["dataset_split"] == "test":
                self.test_records.append(record)
            else:
                raise ValueError("Invalid split value {} in record: {}".format(record["dataset_split"], record["subject"]))
            
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
        per_label_weights = 1.0 / label_counts.float()

        weights = torch.zeros(size = true_labels.size())

        for i in range(len(label_counts)):
            weights[true_labels == i] = per_label_weights[i] / len(label_counts)

        weights = weights.tolist()
    

        sampler = torch.utils.data.WeightedRandomSampler(weights, 2*len(dataset), replacement = True)
        return sampler


    def train_dataloader(self):
        if self.weighted_sample: 
            sampler = self._get_sampler(self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler = sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle = True, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = True, num_workers = self.num_workers)


