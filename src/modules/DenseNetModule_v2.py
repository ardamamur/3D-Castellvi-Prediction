from __future__ import annotations
import argparse
import typing
import warnings

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics.functional as mF

from models.DenseNet3D import monai_dense169_3d
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt

import torch
torch.autograd.set_detect_anomaly(True)

# Your code here



# Dense Net for Classification
class DenseNetV2(pl.LightningModule):
    def __init__(self, opt, num_classes: int, data_size: tuple, data_channel: int):
        super().__init__()
        self.save_hyperparameters() # this call saves the arguments as hyperparameters
        self.counter = 0
        self.thread = None

        self.opt = opt
        self.val_step_outputs = []
        self.training_step_outputs = []
        self.num_classes = num_classes
        self.n_epoch = opt.n_epochs
        self.scheduler_name = opt.scheduler
        self.optimizer_name = opt.optimizer
        self.data_size = data_size  # 2d
        self.data_channel = data_channel

        self.network_id = opt.model
        self.network = self.networkX(self.network_id, pretrained=False)

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.softmax = nn.Softmax(dim=1)

        self.l2_reg_w = 0.0
        print(f"{self._get_name()} loaded with {self.num_classes} classes, data_size {self.data_size} and {self.data_channel} channel")

    def forward(self, x):
        logits = self.network(x)  # [-1, 1]
        return logits


    def training_step(self, batch, batch_idx):
        #target = batch["target"]
        results = self.calc_pred(batch, detach2cpu=False)
        # Calculate 2 loss for 2 different output
        loss1 = results["loss1"]
        loss2 = results["loss2"]
        loss = loss1 + loss2 # Total loss is the sum of the two task losses
        self.training_step_outputs.append(results)
        print(f"Training Loss 1: {loss1.item():.4f}")
        print(f"Training Loss 2: {loss2.item():.4f}")
        print(f"Training Loss Total: {loss.item():.4f}")
        return {"loss": loss}  # return the total loss here



    def on_train_epoch_end(self) -> None:
        if self.training_step_outputs is not None:
            metrics = self.cat_metrics(self.training_step_outputs)
            
            avg_loss1 = metrics[0].mean()
            avg_loss2 = metrics[1].mean()
            avg_loss = avg_loss1 + avg_loss2

            self.log("train_loss", avg_loss1, on_epoch=True)
            # self.log("train_loss1", avg_loss1, on_epoch=True)
            # self.log("train_loss2", avg_loss2, on_epoch=True)

            pred_cls1_a = metrics[4]
            pred_cls2_a = metrics[5]
            gt_cls1_a = metrics[6]
            gt_cls2_a = metrics[7]

            metrics_data = [(pred_cls1_a, gt_cls1_a), (pred_cls2_a, gt_cls2_a)]
            
            for i, (pred, gt) in enumerate(metrics_data, start=1):
                matthews_acc = mF.matthews_corrcoef(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                acc = mF.cohen_kappa(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                f1score = mF.f1_score(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                
                self.logger.experiment.add_scalar(f'train_matthews_acc_{i}', matthews_acc, self.current_epoch)
                self.logger.experiment.add_scalar(f"train_cohen_acc_{i}", acc, self.current_epoch)
                self.logger.experiment.add_scalar(f"train_f1score_{i}", f1score, self.current_epoch)

            self.training_step_outputs = []  # Clear the outputs at the end of each epoch


    def configure_optimizers(self) -> dict:
        
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay),
        }
        optimizer = optimizer_dict[self.optimizer_name]
        lr_scheduler = self.init_lr_scheduler(self.scheduler_name, optimizer)
        if lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                },
            }
        return {"optimizer": optimizer}

    def loss_function(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        loss = self.cross_entropy(logits, labels)
        return loss  # Return two separate losses
        # if self.l2_reg_w > 0.0:
        #     l2_reg = torch.tensor(0.0, device=self.device).to(non_blocking=True)
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param).to(self.device, non_blocking=True)
        #     loss += l2_reg * self.l2_reg_w
        # return loss


    def calc_pred(self, batch, detach2cpu: bool = False) -> dict:
        target = batch["target"]
        gt_cls = batch["class"]
        return self.calc_pred_tensor(target, gt_cls, detach2cpu=detach2cpu)

    def calc_pred_tensor(self, target, gt_cls, detach2cpu: bool = False) -> dict:

        logits1, logits2 = self.forward(target)
        gt_cls1, gt_cls2 = gt_cls[0], gt_cls[1]
        loss1, loss2 = self.loss_function(logits1, gt_cls1), self.loss_function(logits2, gt_cls2)
        
        with torch.no_grad():
            pred_x1, pred_x2 = self.softmax(logits1), self.softmax(logits2)
            _, pred_cls1 = torch.max(pred_x1, 1)
            _, pred_cls2 = torch.max(pred_x2, 1)
        
            if detach2cpu:
                gt_cls1, gt_cls2 = gt_cls1.detach().cpu(), gt_cls2.detach().cpu()
                pred_x1, pred_x2 = pred_x1.detach().cpu(), pred_x2.detach().cpu()
                pred_cls1, pred_cls2 = pred_cls1.detach().cpu(), pred_cls2.detach().cpu()


        result = {"loss1": loss1, "loss2": loss2, 
                  "pred_x1": pred_x1, "pred_cls1": pred_cls1, "gt_cls1": gt_cls1,
                  "pred_x2": pred_x2, "pred_cls2": pred_cls2, "gt_cls2": gt_cls2}
        
        return result

    def validation_step(self, batch, batch_idx):
        result = self.calc_pred(batch, detach2cpu=True)
        self.val_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        if self.val_step_outputs is not None:
            metrics = self.cat_metrics(self.val_step_outputs)

            avg_loss1 = metrics[0].mean()
            avg_loss2 = metrics[1].mean()
            avg_loss = avg_loss1 + avg_loss2

            #self.log("val_loss1", avg_loss1, on_epoch=True)
            #self.log("val_loss2", avg_loss2, on_epoch=True)
            self.log("val_loss", avg_loss, on_epoch=True)

            pred_cls1_a = metrics[4]
            pred_cls2_a = metrics[5]
            gt_cls1_a = metrics[6]
            gt_cls2_a = metrics[7]

            metrics_data = [(pred_cls1_a, gt_cls1_a), (pred_cls2_a, gt_cls2_a)]
            
            for i, (pred, gt) in enumerate(metrics_data, start=1):
                matthews_acc = mF.matthews_corrcoef(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                acc = mF.cohen_kappa(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                f1score = mF.f1_score(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')
                f1_p_cls = mF.f1_score(preds=pred, target=gt, average="none", num_classes=self.num_classes, task='multiclass')
                prec = mF.precision(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')

                confmat = mF.confusion_matrix(preds=pred, target=gt, num_classes=self.num_classes, task='multiclass')

                try:
                    import seaborn as sns
                    df_cm = pd.DataFrame(confmat.numpy(), index=range(self.num_classes), columns=range(self.num_classes))
                    plt.figure(figsize=(10, 7))
                    fig_ = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Spectral").get_figure()
                    self.logger.experiment.add_figure(f"Confusion matrix_{i}", fig_, self.current_epoch)
                    plt.close(fig_)
                except Exception as e:
                    print("caught exception in confusion matrix", e)

                # Log important validation values
                self.logger.experiment.add_scalar(f'val_matthews_acc_{i}', matthews_acc, self.current_epoch)
                self.logger.experiment.add_scalar(f"val_cohen_acc_{i}", acc, self.current_epoch)
                self.logger.experiment.add_scalar(f"val_f1score_{i}", f1score, self.current_epoch)
                self.logger.experiment.add_scalar(f"val_prec_{i}", prec, self.current_epoch)

                self.logger.experiment.add_text(f"f1_p_cls_{i}", str(f1_p_cls.tolist()), self.current_epoch)

            self.val_step_outputs = []


    @torch.no_grad()
    def cat_metrics(self, outputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss1_a, loss2_a = [], []
        pred_x1_a, pred_x2_a = [], []
        pred_cls1_a, pred_cls2_a = [], []
        gt_cls1_a, gt_cls2_a = [], []
        
        for o in outputs:
            loss1_a.append(o["loss1"])
            loss2_a.append(o["loss2"])
            pred_x1_a.append(o["pred_x1"])
            pred_x2_a.append(o["pred_x2"])
            pred_cls1_a.append(o["pred_cls1"])
            pred_cls2_a.append(o["pred_cls2"])
            gt_cls1_a.append(o["gt_cls1"])
            gt_cls2_a.append(o["gt_cls2"])
        
        return (torch.Tensor(loss1_a),
                torch.Tensor(loss2_a),
                torch.cat(pred_x1_a),
                torch.cat(pred_x2_a),
                torch.cat(pred_cls1_a).to(torch.int),
                torch.cat(pred_cls2_a).to(torch.int),
                torch.cat(gt_cls1_a).to(torch.int),
                torch.cat(gt_cls2_a).to(torch.int)
                )

    def init_lr_scheduler(self, name, optimizer):
        scheduler_dict = {
            "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch, eta_min=1e-7),
            "exponential": lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95),
            "ReduceLROnPlateau" : lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True),
            "polynomial": lr_scheduler.PolynomialLR(optimizer, total_iters=self.opt.total_iterations, power=0.9),
        }
        if name in scheduler_dict:
            return scheduler_dict[name]
        return None

    def networkX(self, id: int | str, pretrained: bool = True, true_3d: bool = True) -> nn.Module:
        """fetches and inits the corresponding network

        Args:
            id: int or str of network

        Returns:
            modified and init network based on internal parameter
        """
        if id == "densenet":
            network_func = monai_dense169_3d
        else:
            raise Exception("Not Implemented")
        
        return network_func(data_channel=self.data_channel, num_classes=self.num_classes, pretrained=pretrained)

    def __str__(self):
        return f"{self.network_id}"
