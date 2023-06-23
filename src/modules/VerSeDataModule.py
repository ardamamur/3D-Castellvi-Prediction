import pytorch_lightning as pl
import pandas as pd
from utils._prepare_data import DataHandler
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader, random_split
import torch
from sklearn.model_selection import KFold
import os

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
        self.num_folds = 5
        self.split_seed = 123

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        records = self.processor.verse_records
        print(f"Total records: {len(records)}")

        if not self.opt.cross_validation:
            if not self.opt.flip_all:
                # # remove recorde if their flip value is 1 and castellvi value does not contain 'a'
                records = [record for record in records if record["flip"] == 1 and (record["castellvi"]!='2a' or record["castellvi"]!='3a')]
                print("----------------------------------------------------------------------------------")
                print(f"Total records after removing non-flipped records: {len(records)}")

            for record in records:
                if record["dataset_split"] == "train":
                    self.train_records.append(record)
                elif record["dataset_split"] == "val":
                    self.val_records.append(record)
                elif record["dataset_split"] == "test":
                    self.val_records.append(record)
                    self.val_records.append(record)
                else:
                    raise ValueError("Invalid split value {} in record: {}".format(record["dataset_split"], record["subject"]))
                
            if stage in {'fit', None}:
                self.train_dataset = VerSe(self.opt, self.processor, self.train_records, training=True)
                self.val_dataset = VerSe(self.opt, self.processor, self.val_records, training=False)
            
            if stage in {'test', None}:
                self.train_dataset = VerSe(self.opt, self.processor, self.test_records, training=False)
        
        else:
            # get only records with split value of train and val
            records = [record for record in records if record["dataset_split"] == "train" or record["dataset_split"] == "val"]
            kf = KFold(n_splits = self.num_folds, shuffle = True, random_state = self.split_seed)

            fold_datasets = []
            count = 0
            for train_indexes, val_indexes in kf.split(records):
                print(count)
                train_records = [records[i] for i in train_indexes]
                val_records = [records[j] for j in val_indexes]
                
                train_records_subject_name = []
                val_records_subject_name = []
                for record in train_records:
                    train_records_subject_name.append([record['subject'], record["flip"]])
                for record in val_records:
                    val_records_subject_name.append([record['subject'] , record["flip"]])

                train_dataset = VerSe(self.opt, self.processor, train_records, training=True)
                val_dataset = VerSe(self.opt, self.processor, val_records, training=False)
                
                # Don't forget to change path
                train_file_name = '/u/home/ank/3D-Castellvi-Prediction/src/dataset/' + "fold" + str(count) + "_train" 
                val_file_name = '/u/home/ank/3D-Castellvi-Prediction/src/dataset/' + "fold" + str(count) + "_val"

                with open(train_file_name, 'w') as fp:
                    for item in train_records_subject_name:
                        fp.write("%s,%s\n" % (item[0],item[1]))
                
                with open(val_file_name, 'w') as fp:
                    for item in val_records_subject_name:
                        fp.write("%s,%s\n" % (item[0],item[1]))
                        

                fold_datasets.append((train_dataset, val_dataset))
                count +=1
                
        

            
            if stage == 'fit' or stage is None:
                self.train_dataset, self.val_dataset = fold_datasets[self.current_fold]
            
            
        
    def set_current_fold(self, fold_index: int):
        
        self.current_fold = fold_index % self.num_folds
        self.setup(stage='fit')



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


