from typing import Dict, Sequence
import pytorch_lightning as pl
import torch
from monai.metrics import MSEMetric, MAEMetric, RMSEMetric, compute_auc_roc

class VerSeLightning(pl.LightningModule):
    def __init__(self) -> None

