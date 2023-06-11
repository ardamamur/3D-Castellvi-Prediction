from monai.networks.nets import ResNet, resnet101
import torch
from typing import Any, Optional, Sequence, Tuple, Type, Union

resnet = resnet101(
    pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3,
)

def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_constructor: callable = resnet,
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
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
    print(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
    print(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
    print(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net, inside
