import warnings

import torchvision
from torchvision import models
import torch
from torch import nn


def monai_dense169_3d(data_channel, num_classes, pretrained):
    from monai.networks.nets import DenseNet169
    # pretrained=False because not supported for 3D
    if pretrained:
        warnings.warn("monia_densenet_3d doesnt support pretrained, is set to False", UserWarning)
    network = DenseNet169(in_channels=1, pretrained=False, spatial_dims=3, out_channels=num_classes)
    return network


