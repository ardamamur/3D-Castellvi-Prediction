
from torch import nn
from torchsummary import summary
import torch

"Based on https://towardsdatascience.com/review-3d-u-net-volumetric-segmentation-medical-image-segmentation-8b592560fac1"

""" 
    In the analysis path, each layer contains two 3 x 3 x 3 convolutions each followed by a ReLU, and then a 2 x 2 x 2 max pooling with strides of two in each dimension.
    In the synthesis path, each layer consists of an up-convolution of 2 x 2 x 2 by strides of two in each dimension, followed by two 3 x 3 x3 convolutions each followed by a ReLU.
    Shortcut connections from layers of equal resolution in the analysis path provide the essential high-resolution features to the synthesis path.
    In the last layer, a 1 x 1 x 1 convolution reduces the number of output channels to the number of labels which is 3.
    Batch normalization before each ReLU."""


class Conv3DBlock(nn.Module):
    """
    
    in_channels -> number of input channels
    out_channels -> desired number of output channels
    bottleneck -> specifies the bottlneck block
    
    """
    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = in_channels, out_channels = out_channels//2, kernel_size = (3, 3, 3), padding = 1)
        self.bn1 = nn.BatchNorm3d(num_features = out_channels//2)
        self.conv2 = nn.Conv3d(in_channels = out_channels//2, out_channels = out_channels, kernel_size=(3, 3, 3), padding = 1)
        self.bn2 = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.ReLU()

        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = 2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    __init__()
    in_channels -> number of input channels
    out_channels -> number of residual connections channels to be concatenated
    last_layer -> specifies the last output layer
    num_classes -> specifies the number of output channels for dispirate classes
    
    forward()
    input -> input Tensor
    residual -> residual connection to be concatenated with input
    """

    def __init__(self, in_channels, res_channels = 0, last_layer = False, num_classes = None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels = in_channels, out_channels = in_channels, kernel_size = (2, 2, 2), stride = 2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features = in_channels//2)
        self.conv1 = nn.Conv3d(in_channels = in_channels + res_channels , out_channels = in_channels//2, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels = in_channels//2, out_channels = in_channels//2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels = in_channels//2, out_channels = num_classes, kernel_size = (1, 1, 1))
            
        
    def forward(self, input, residual = None):
        out = self.upconv1(input)
        if residual != None: 
            out = torch.cat((out, residual), 1)
        
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: 
            out = self.conv3(out)
        return out
        



class UNet3D(nn.Module):
    """
    The 3D UNet model
    __init__()
    in_channels -> number of input channels
    num_classes -> specifies the number of output channels or masks for different classes
    level_channels -> the number of channels at each level (count top-down)
    bottleneck_channel -> the number of bottleneck channels 
    device -> the device on which to run the model

    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out



if __name__ == '__main__':
    model = UNet3D(in_channels=3, num_classes=2)
    summary(model=model, input_size=(3, 128,86,136), batch_size=-1)
