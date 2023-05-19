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


# region DENSENET
def dense169(data_channel, num_classes, pretrained):
    network = models.densenet169(pretrained=pretrained)
    setup_densenet_layer(network, data_channel, num_classes, 64, 1664)
    return network


def dense121(data_channel, num_classes, pretrained):
    network = models.densenet121(pretrained=pretrained)  # (weights=models.DenseNet121_Weights.DEFAULT)
    setup_densenet_layer(network, data_channel, num_classes, 64, 1024)
    return network


def dense161(data_channel, num_classes, pretrained):
    network = models.densenet161(pretrained=pretrained)  # (weights=models.DenseNet161_Weights.DEFAULT)
    setup_densenet_layer(network, data_channel, num_classes, 96, 2208)
    return network


def dense201(data_channel, num_classes, pretrained):
    network = models.densenet201(pretrained=pretrained)  # (weights=models.DenseNet201_Weights.DEFAULT)
    setup_densenet_layer(network, data_channel, num_classes, 64, 1920)
    return network
#endregion

#region RESNET
def resnet18(data_channel, num_classes, pretrained):
    network = models.resnet18(pretrained=pretrained)  # (weights=models.ResNet18_Weights.DEFAULT)
    setup_resnet_layer(network, data_channel, num_classes, 64, 512)
    return network


def resnet34(data_channel, num_classes, pretrained):
    network = models.resnet34(pretrained=pretrained)  # (weights=models.ResNet34_Weights.DEFAULT)
    setup_resnet_layer(network, data_channel, num_classes, 64, 512)
    return network


def resnet50(data_channel, num_classes, pretrained):
    network = models.resnet50(pretrained=pretrained)  # (weights=models.ResNet50_Weights.DEFAULT)
    setup_resnet_layer(network, data_channel, num_classes, 64, 2048)
    return network


def resnet101(data_channel, num_classes, pretrained):
    network = models.resnet101(pretrained=pretrained)  # (weights=models.ResNet101_Weights.DEFAULT)
    setup_resnet_layer(network, data_channel, num_classes, 64, 2048)
    return network


def resnet152(data_channel, num_classes, pretrained):
    network = models.resnet152(pretrained=pretrained)  # (weights=models.ResNet152_Weights.DEFAULT)
    setup_resnet_layer(network, data_channel, num_classes, 64, 2048)
    return network
#endregion


# FOR SETUP
def setup_resnet_layer(network, data_channel, num_classes, conv1_out, classifier):
    network.conv1 = torch.nn.Conv2d(data_channel, conv1_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    network.fc = nn.Linear(classifier, num_classes)


def setup_densenet_layer(network, data_channel, num_classes, conv1_out, classifier):
    network.features[0] = torch.nn.Conv2d(data_channel, conv1_out, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    network.classifier = nn.Linear(classifier, num_classes)