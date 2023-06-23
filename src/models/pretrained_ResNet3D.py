from monai.networks.nets import EfficientNetBN, ResNet, resnet18, resnet50, resnet101, resnet152
import torch
from typing import Any, Optional, Sequence, Tuple, Type, Union

model_constructor = { 
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

weight_constructor = {
    'resnet18': '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/pretrained_weigths/resnet/resnet_18.pth',
    'resnet50': '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/pretrained_weigths/resnet/resnet_50.pth',
    'resnet101': '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/pretrained_weigths/resnet/resnet_101.pth',
    'resnet152': '/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/pretrained_weigths/resnet/resnet_152.pth',
}

def create_pretrained_medical_resnet(
    model_type: str = 'resnet18',
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 1,
    **kwargs_monai_resnet: Any
) -> Tuple[ResNet, Sequence[str]]:
    """This si specific constructor for MONAI ResNet module loading MedicalNEt weights.

    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """


    net = model_constructor[model_type](
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )

    # Adjust the final layer to match the number of classes
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # Load your pre-trained weights
    pretrained_model = weight_constructor[model_type] 
    try:
        pretrained_state = torch.load(pretrained_model)['state_dict']  
        net_state = net.state_dict()

        for name, param in pretrained_state.items():
            if name not in net_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            net_state[name].copy_(param)

    except Exception as error:
        print(f'Error loading the pretrained model: {error}')
        print('The model will be initialized with random weights.')

    # Freeze all layers
    for param in net.parameters():
        param.requires_grad = False

    # Unfreeze the top layers
    for param in net.fc.parameters():
        param.requires_grad = True

    return net







































# if not unfreeze_top:
#     net_dict = net.state_dict()
#     pretrain = torch.load(pretrained_path)
#     pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
#     missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
#     print(f"missing in pretrained: {len(missing)}")
#     inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
#     print(f"inside pretrained: {len(inside)}")
#     unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
#     print(f"unused pretrained: {len(unused)}")
#     assert len(inside) > len(missing)
#     assert len(inside) > len(unused)

#     pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
#     net.load_state_dict(pretrain['state_dict'], strict=False)
#     return net, inside

# else:
#     net_dict = net.state_dict()
#     pretrain = torch.load(pretrained_path)
#     pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    
#     pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
#     net.load_state_dict(pretrain['state_dict'], strict=False)

#     # freeze all layers
#     for param in net.parameters():
#         param.requires_grad = True

#     # unfreeze the last ConvBlock and FC layers
#     for param in net.layer4.parameters():
#         param.requires_grad = True
#     for param in net.fc.parameters():
#         param.requires_grad = True

#     return net

