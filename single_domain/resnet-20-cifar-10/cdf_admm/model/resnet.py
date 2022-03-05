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


from utils.admm import ADMM

from .quantization_uniform import *


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
    
    if self.training:
        dim = args.train_batch_size
    else:
        dim = args.eval_batch_size
        
    self.admm0 = ADMM(dim).to(device)
    self.admm1 = ADMM(dim).to(device)
    
    
    self.act_q0 = activation_quantize_fn(a_bit = abit, stage = stage, admm = self.admm0)
    self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage, admm = self.admm1)
    
    self.bn0 = nn.BatchNorm2d(out_planes)
    self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    
    
    self.skip_conv = None
    if stride != 1:
      self.admm_skip = ADMM(dim).to(device)
      self.act_skip_q = activation_quantize_fn(a_bit = abit, stage = stage, admm = self.admm_skip)

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
    trans_loss = 0.
    if self.skip_conv is not None:
      shortcut = self.skip_conv(x)
      shortcut, loss = self.act_skip_q(self.skip_bn(shortcut))
      trans_loss += loss
    else:
      shortcut = x
    
    out = self.conv0(x)
    out, loss = self.act_q0(self.bn0(out))
    trans_loss += loss
    out = F.relu(out)
    
    out = self.conv1(out)
    out, loss = self.act_q1(self.bn1(out))
    trans_loss += loss
   
    out += shortcut
    out = F.relu(out)
    return out, trans_loss


class PreActResNet(nn.Module):
  def __init__(self, block, num_units, wbit, abit, stage, num_classes, block_bits = None):
    super(PreActResNet, self).__init__()
    
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    self.conv0 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

    if self.training:
        dim = args.train_batch_size
    else:
        dim = args.eval_batch_size
    
    self.admm0 = ADMM(dim).to(device)
    
    self.act_q0 = activation_quantize_fn(a_bit = abit, stage = stage, admm = self.admm0)
    
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
    trans_loss = 0.
    out = self.conv0(x)
    out = self.bn(out)
    out, loss = self.act_q0(out)
    trans_loss += loss
    out = F.relu(out)
    
    for layer in self.layers:
      out, loss = layer(out)
      trans_loss += loss
    
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.logit(out)
    return out, trans_loss


def resnet20_quant(bitW, abitW, stage, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], bitW, abitW, stage, num_classes=num_classes)

def resnet56_quant(bitW, abitW, stage, num_classes=10):
  return PreActResNet(PreActBlock_conv_Q, [9, 9, 9], bitW, abitW, stage, num_classes=num_classes)


if __name__ == '__main__':
  features = []


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    features.append(output.data.cpu().numpy())


  net = resnet20(wbits=1, abits=2)
  for m in net.modules():
    m.register_forward_hook(hook)

  y = net(torch.randn(1, 3, 32, 32))
  print(y.size())

def resnet20_cifar(bitW=1):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], bitW=bitW)
    return model

def resnet20_stl(bitW=1):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], bitW=bitW, first_stride=3)
    return model


def resnet32_cifar(bitW=1):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], bitW=bitW)
    return model


def resnet44_cifar(bitW=1):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], bitW=bitW)
    return model

'''
def resnet56_quant(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model
'''
def resnet_56(**kwargs):
    return ResNet(ResBasicBlock, 56, **kwargs)


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

#if __name__ == '__main__':
    ## net = preact_resnet110_cifar()
    #net = resnet56_quant(bitW=4)
    #y = net(torch.zeros(1, 3, 32, 32), 'myQ')
    ##print(net)
    ##print(y.size())
