import pytorch_lightning as pl
import pandas as pd
from utils._prepare_data import DataHandler
from dataset.VerSe import VerSe
from torch.utils.data import DataLoader

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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


