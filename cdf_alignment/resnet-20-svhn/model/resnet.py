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

from .quantization import *


device = torch.device(f"cuda:{args.gpus[0]}")


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class PreActBlock_conv_Q(nn.Module):
  '''Pre-activation version of the BasicBlock.'''

  def __init__(self, stage, wbit, abit, in_planes, out_planes, stride=1):
    super(PreActBlock_conv_Q, self).__init__()
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    self.act_q0 = activation_quantize_fn(a_bit = abit, stage = stage)
    self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage)
    self.act_skip_q = activation_quantize_fn(a_bit = abit, stage = stage)

    self.bn0 = nn.BatchNorm2d(out_planes)
    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    
    
    self.skip_conv = None
    if stride != 1:
      self.skip_conv = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
      
      self.skip_bn = nn.BatchNorm2d(out_planes)

    #self.alpha = nn.Parameter(torch.rand(1)) 
    '''
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != out_planes:
        self.shortcut = LambdaLayer(lambda x: F.pad(
                x[:, :, ::2, ::2], (0, 0, 0, 0, out_planes//4, out_planes//4),
                "constant", 0))
    '''
  def forward(self, x):
   
    if self.skip_conv is not None:
      shortcut = self.skip_conv(x)
      shortcut = self.act_skip_q(self.skip_bn(shortcut))
    else:
      shortcut = x
    
    out = self.conv0(x)
    out = F.relu(self.act_q0(self.bn0(out)))
    
    out = self.conv1(out)
    out = self.act_q1(self.bn1(out))
   
    out += shortcut
    out = F.relu(out)
    return out


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, wbit, abit, stage, num_classes, block_bits = None):
    super(PreActResNet, self).__init__()
    
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    self.conv0 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    #self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.act_q0 = activation_quantize_fn(a_bit = abit, stage = stage)
    
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
    
    #Linear = linear_Q_fn(32, stage)
    #self.logit = Linear(64, num_classes)
    self.logit = nn.Linear(64, num_classes)


  def forward(self, x):
    out = self.conv0(x)
    out = self.bn(out)
    out = self.act_q0(out)
    out = F.relu(out)
    
    for layer in self.layers:
      out = layer(out)
    
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.logit(out)
    return out


def resnet20_quant(bitW, abitW, stage, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], bitW, abitW, stage, num_classes=num_classes)

def resnet56_quant(bitW, abitW, stage, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [9, 9, 9], bitW, abitW, stage, num_classes=num_classes)



