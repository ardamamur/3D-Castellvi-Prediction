import pytorch_lightning as pl
from typing import Dict, Sequence
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet3D import *

from utils._get_model import _generate_model, _get_num_classes

class ResNetLightning(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        
        # TODO add another loss function for multi class classification
        # matthew's correlation coefficient as eval metrics
        # In classification task you can think as regression task. 
        self.criterion = nn.BCELoss() # or use another suitable loss function
        self.learning_rate = params.learning_rate
        self.weight_decay = params.weight_decay
        self.total_iterations = params.total_iterations
        self.num_classes = _get_num_classes(
            binary_classification=params.binary_classification,
            castellvi_classes=params.castellvi_classes)
        self.model = _generate_model(
            model=params.model,
            num_classes=self.num_classes,
            no_cuda=params.no_cuda)

        print(self.model.parameters())
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x))
        loss = self.criterion(y_hat, y)
        self.log('loss', loss, on_step=True, on_epoch=True)
        print(f'Train loss: {loss}')
        return {'loss':loss}

    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x))
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss)
        print(f'Val loss: {val_loss}')
        return {'val_loss':val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self.model(x))
        test_loss = self.criterion(y_hat, y)
        self.log('test_loss', test_loss)
        return {'test_loss':test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.total_iterations, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
