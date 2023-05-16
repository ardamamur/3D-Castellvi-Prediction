import torch
from torch import nn
from models.ResNet3D import *

def generate_model(opt,
                   input_shape:tuple,  # (W, H, D)
                   num_classes:int
                ):
    
    if opt.binary_classification:
        num_classes = 1
    else:
        num_classes = len(opt.castellvi_classes) 

    if 'resnet' in opt.model:
        if '10' in opt.model:
            model = resnet10(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '18' in opt.model:
            model = resnet18(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '34' in opt.model:
            model = resnet34(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '50' in opt.model:
            model = resnet50(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '101' in opt.model:
            model = resnet101(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '152' in opt.model:
            model = resnet152(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        elif '200' in opt.model:
            model = resnet200(
                sample_input_W=input_shape[0],
                sample_input_H=input_shape[1],
                sample_input_D=input_shape[2],
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_classes=num_classes)
        else:
            raise Exception('Not Implemented')
        
        return model
    else:
        pass

    