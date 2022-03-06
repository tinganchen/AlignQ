import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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
    
    
class Block(nn.Module):
    def __init__(self, stage, wbit, abit, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        
        Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
        self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q2 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q3 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_skip = activation_quantize_fn(a_bit = abit, stage = stage)
        
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6()

        self.shortcut = None
        if stride == 1:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
                self.act_skip,
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_q1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_q2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act_q3(out)
        
        if self.stride == 1:
            out += self.shortcut(x)
	
        return out


class MobileNetV2(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    

    def __init__(self, block, wbit, abit, stage, num_classes=10):
        
        super(MobileNetV2, self).__init__()
        
        Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
        self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q2 = activation_quantize_fn(a_bit = abit, stage = stage)
        
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, wbit, abit, stage, in_planes=32)
        self.conv2 = Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace = True)
        
        self.avg_pool2d = nn.AvgPool2d(4)
        
        
    def _make_layers(self, block, wbit, abit, stage, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(stage, wbit, abit, in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_q1(out)
        out = self.relu(out)
        
        out = self.layers(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_q2(out)
        out = self.relu(out)

        self.out = self.avg_pool2d(out)
        out = self.out.view(self.out.size(0), -1)
        out = self.linear(out)
        return out


    
def mobile_v2(wbit, abit, stage, **kwargs):
    return MobileNetV2(Block, wbit, abit, stage, **kwargs)


    