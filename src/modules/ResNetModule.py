from __future__ import annotations
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics.functional as mF
from models.pretrained_ResNet3D import *
from models.ResNet3D import *
from utils._get_model import _get_weights
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt


class ResNet(pl.LightningModule):
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
        
        self.softmax = nn.Softmax(dim=1)

        # TODO : Update masterlist parameter
        if opt.weighted_loss:
            weights = _get_weights(opt.master_list , rigth_side=True)
            weights = torch.tensor(weights).cuda()
            self.cross_entropy = nn.CrossEntropyLoss(weight=weights, reduction="mean")
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")


        self.l2_reg_w = 0.0
        print(f"{self._get_name()} loaded with {self.num_classes} classes, data_size {self.data_size} and {self.data_channel} channel")
    
    
    def forward(self, x):
        x = x.float()
        logits = self.network(x)  # [-1, 1]
        return logits
    

    def training_step(self, batch, batch_idx):

        target = batch["target"]
        result = self.calc_pred(batch, detach2cpu=False)
        loss = result["loss"]
        self.training_step_outputs.append(result)
        #self.logger.info(f"Training Loss: {loss:.4f}")
        print(f"Training Loss: {loss.item():.4f}")
        return {"loss": loss}  # return the loss here, we will use it later


    def on_train_epoch_end(self) -> None:
        if self.training_step_outputs is not None:

            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()  # Calculate average loss over the epoch
            self.log("train_loss", avg_loss, on_epoch=True)  # Log the average loss
            loss_a, predictions_a, pred_cls_a, gt_cls_a = self.cat_metrics(self.training_step_outputs)
            matthews_acc = mF.matthews_corrcoef(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            acc = mF.cohen_kappa(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            f1score = mF.f1_score(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            self.logger.experiment.add_scalar('train_matthews_acc', matthews_acc, self.current_epoch)
            self.logger.experiment.add_scalar("train_cohen_acc", acc, self.current_epoch)
            self.logger.experiment.add_scalar("train_f1score", f1score, self.current_epoch)
            self.training_step_outputs = []  # Clear the outputs at the end of each epoch

        if self.opt.model == 'pretrained_resnet' and self.opt.gradual_freezing:
            # Gradual Freezing 

            # A common approach is to start by training only the last layers for a certain number of epochs, 
            # then gradually unfreeze earlier layers and continue training. For instance, you might do something like this:

            # Train only the last layer for 20 epochs.
            # Unfreeze and train the last two layers for the next 20 epochs.
            # Unfreeze and train the last three layers for the next 20 epochs.
            # And so on...
            # Ultimately, the schedule for layer unfreezing is a hyperparameter of your training process 
            # that you may need to tune based on your specific task.

            # Get the current epoch
            epoch = self.current_epoch

            # Specify the layers to unfreeze at each epoch
            layers_to_unfreeze = {
                30: ["layer4", "layer3"],
                60: ["layer2", "layer1"],
                90: ["bn1", "relu", "maxpool", "conv1"],
            }

            # Unfreeze layers
            if epoch in layers_to_unfreeze:
                for layer_name in layers_to_unfreeze[epoch]:
                    for name, child in self.network.named_children():
                        if name == layer_name:
                            print(f"Unfreezing {name}")
                            for param in child.parameters():
                                param.requires_grad = True


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
        # if self.l2_reg_w > 0.0:
        #     l2_reg = torch.tensor(0.0, device=self.device).to(non_blocking=True)
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param).to(self.device, non_blocking=True)
        #     loss += l2_reg * self.l2_reg_w
        # return loss
        return loss


    def calc_pred(self, batch, detach2cpu: bool = False) -> dict:
        target = batch["target"]
        gt_cls = batch["class"]
        return self.calc_pred_tensor(target, gt_cls, detach2cpu=detach2cpu)


    def calc_pred_tensor(self, target, gt_cls, detach2cpu: bool = False) -> dict:
        logits = self.forward(target)
        print(len(gt_cls))
        print(gt_cls)
        assert (gt_cls >= 0).all() and (gt_cls < self.num_classes).all(), "Labels out of range"
        loss = self.loss_function(logits, gt_cls)

        with torch.no_grad():
            pred_x = self.softmax(logits)  # , dim=1)
            _, pred_cls = torch.max(pred_x, 1)
            if detach2cpu:
                # From here on CPU
                gt_cls = gt_cls.detach().cpu()
                pred_x = pred_x.detach().cpu()
                pred_cls = pred_cls.detach().cpu()
        result = {"loss": loss, "pred_x": pred_x, "pred_cls": pred_cls, "gt_cls": gt_cls}
        return result


    def validation_step(self, batch, batch_idx):
        result = self.calc_pred(batch, detach2cpu=True)
        self.val_step_outputs.append(result)
        return result


    def on_validation_epoch_end(self):
        if self.val_step_outputs is not None:
            loss_a, predictions_a, pred_cls_a, gt_cls_a = self.cat_metrics(self.val_step_outputs)
            loss = torch.mean(loss_a)
            self.log("val_loss", loss, on_epoch=True)
            
            matthews_acc = mF.matthews_corrcoef(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            acc = mF.cohen_kappa(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            f1score = mF.f1_score(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            f1_p_cls = mF.f1_score(preds=pred_cls_a, target=gt_cls_a, average="none", num_classes=self.num_classes, task='multiclass')
            prec = mF.precision(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            confmat = mF.confusion_matrix(preds=pred_cls_a, target=gt_cls_a, num_classes=self.num_classes, task='multiclass')
            
            try:
                import seaborn as sns
                df_cm = pd.DataFrame(confmat.numpy(), index=range(self.num_classes), columns=range(self.num_classes))
                plt.figure(figsize=(10, 7))
                fig_ = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Spectral").get_figure()
                self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
                plt.close(fig_)
            except Exception as e:
                print("caught exception in confusion matrix", e)

            # Log important validation values
            self.logger.experiment.add_scalar('train_matthews_acc', matthews_acc, self.current_epoch)
            self.logger.experiment.add_scalar('val_matthews_acc', matthews_acc, self.current_epoch)
            self.logger.experiment.add_scalar("val_cohen_acc", acc, self.current_epoch)
            self.logger.experiment.add_scalar("val_f1score", f1score, self.current_epoch)
            self.logger.experiment.add_scalar("val_prec", prec, self.current_epoch)
            self.logger.experiment.add_text("f1_p_cls", str(f1_p_cls.tolist()), self.current_epoch)
            self.val_step_outputs = []

    @torch.no_grad()
    def cat_metrics(self, outputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_a: list[float] = []
        predictions_a: list[torch.Tensor] = []
        pred_cls_a: list[torch.Tensor] = []
        gt_cls_a: list[torch.Tensor] = []
        for o in outputs:
            loss, pred_x, pred_cls, vert_cls = o["loss"], o["pred_x"], o["pred_cls"], o["gt_cls"]
            # loss_a.append(loss)
            loss_a.append(loss)
            predictions_a.append(pred_x)
            pred_cls_a.append(pred_cls)
            gt_cls_a.append(vert_cls)
        loss_t = torch.Tensor(loss_a)
        predictions_t = torch.cat(predictions_a)
        pred_cls_t = torch.cat(pred_cls_a).to(torch.int)
        gt_cls_t = torch.cat(gt_cls_a).to(torch.int)
        return loss_t, predictions_t, pred_cls_t, gt_cls_t


    def init_lr_scheduler(self, name, optimizer):
        scheduler_dict = {
            "cosine": lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epoch, eta_min=1e-7),
            "exponential": lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95),
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
        if id=='resnet':
            return resnet18(pretrained=False,
                                n_input_channels=1,
                                spatial_dims=3,
                                num_classes=self.num_classes)

        elif id == "pretrained_resnet":
            # todo: add pretrained
            # Load the model (default resnet18)
            net = create_pretrained_medical_resnet(model_type=self.opt.model_type,
                                                    spatial_dims=3,
                                                    n_input_channels=1,
                                                    num_classes=self.num_classes)
            # for n, param in net.named_parameters():
            #     param.requires_grad = bool(n not in pretraineds_layers)
            # return net
            return net
                        
    def __str__(self):
        return f"{self.network_id}"