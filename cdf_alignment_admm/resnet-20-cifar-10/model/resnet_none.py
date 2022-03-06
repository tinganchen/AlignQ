'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math

import torch.nn.functional as F

import time
import numpy as np

from utils.options import args


from .quantization_apot import * # APOT
#from .quantization_lsq import * # LSQ

device = torch.device(f"cuda:{args.gpus[0]}")

  
class PreActBlock_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''

  def __init__(self, stage, wbit, abit, in_planes, out_planes, stride=1):
    super(PreActBlock_conv_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    
    self.bn0 = nn.BatchNorm2d(out_planes)
    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    
    
    self.skip_conv = None
    if stride != 1:
      self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      
      self.skip_bn = nn.BatchNorm2d(out_planes)

 
  def forward(self, x):
   
    if self.skip_conv is not None:
      shortcut = self.skip_conv(x)
      shortcut = self.skip_bn(shortcut)
    else:
      shortcut = x
    
    out = self.conv0(x)
    out = F.relu(self.bn0(out))
    
    out = self.conv1(out)
    out = self.bn1(out)
   
    out += shortcut
    out = F.relu(out)
    return out


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, wbit, abit, stage, num_classes, block_bits = None):
    super(PreActResNet, self).__init__()
    
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    self.conv0 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    #self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    self.layers = nn.ModuleList()
    in_planes = 16
    strides = [1] * (num_units[0]) + \
              [2] + [1] * (num_units[1] - 1) + \
              [2] + [1] * (num_units[2] - 1)
    channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
    
    if block_bits is None:
        for stride, channel in zip(strides, channels):
          self.layers.append(block(stage, wbit, abit, in_planes, channel, stride))
          in_planes = channel
    else:
        COUNT = 0
        for stride, channel in zip(strides, channels):
          wbit = block_bits[COUNT]
          self.layers.append(block(stage, wbit, abit, in_planes, channel, stride))
          in_planes = channel
          COUNT += 1

    self.bn = nn.BatchNorm2d(16)

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    self.logit = nn.Linear(64, num_classes)
    

  def forward(self, x):
    out = self.conv0(x)
    out = self.bn(out)
    out = F.relu(out)
    
    for layer in self.layers:
      out = layer(out)
    
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.logit(out)
    return out

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def Quantconv3x3(w_bit, a_bit, stage, in_planes, out_planes, stride=1):
    " 3x3 quantized convolution with padding "
    return conv2d_Q_fn(w_bit, a_bit, stage)(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, w_bit, a_bit, stage, inplanes, planes, stride=1, downsample=None, float=False):
        super(BasicBlock, self).__init__()
        if float:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv1 = Quantconv3x3(w_bit, a_bit, stage, inplanes, planes, stride)
            self.conv2 = Quantconv3x3(w_bit, a_bit, stage, planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.expansion = 1

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, w_bit, a_bit, stage, num_classes=10, float=False):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.expansion = 1
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.stage = stage
                
        self.conv1 = Quantconv3x3(w_bit, a_bit, stage, 3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], float=float)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, float=float)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, float=float)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, float=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv2d_Q_fn(self.w_bit, self.a_bit, self.stage)(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
                if float is False else nn.Conv2d(self.inplanes, planes * self.expansion, kernel_size=1,
                                                 stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

        layers = []
        layers.append(block(self.w_bit, self.a_bit, self.stage, self.inplanes, planes, stride, downsample, float=float))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.w_bit, self.a_bit, self.stage, self.inplanes, planes, float=float))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, conv2d_Q_fn):
                m.show_params()



def resnet20_quant(bitW, abitW, stage, **kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], bitW, abitW, stage, **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    pass
    # net = resnet20_cifar(float=True)
    # y = net(torch.randn(1, 3, 64, 64))
    # print(net)
    # print(y.size())

