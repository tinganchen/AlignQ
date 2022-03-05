import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from enum import Enum
from torch.nn.parameter import Parameter

import time

import numpy as np

from utils.options import args

device = torch.device(f"cuda:{args.gpus[0]}")

class LSQ_Quantizer(torch.nn.Module):
    def __init__(self, bits, is_activation=False):
        super(LSQ_Quantizer, self).__init__()

        self.bits = bits

        if(is_activation):
            self.Qn = 0
            self.Qp = 2 ** bits - 1
        else:
            self.Qn = -2**(bits - 1)
            self.Qp = 2 ** (bits - 1) - 1

        self.s = torch.nn.Parameter(torch.Tensor([1.0]))

    def init_step_size(self, x):
        self.s = torch.nn.Parameter(
            x.detach().abs().mean() * 2 / (self.Qp) ** 0.5)

    def grad_scale(self, x, scale):
        y_out = x
        y_grad = x * scale

        y = y_out.detach() - y_grad.detach() + y_grad

        return y

    def round_pass(self, x):
        y_out = x.round()
        y_grad = x
        y = y_out.detach() - y_grad.detach() + y_grad

        return y

    def forward(self, x):
        scale_factor = 1 / (x.numel() * self.Qp) ** 0.5

        scale = self.grad_scale(self.s, scale_factor)
        x = x / scale
        x = x.clamp(self.Qn, self.Qp)

        x_bar = self.round_pass(x)

        x_hat = x_bar * scale

        return x_hat
    
    
def conv2d_Q_fn(w_bit, a_bit, stage):
    class LSQ_Conv2D(torch.nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True):
            super(LSQ_Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                groups, bias)
    
            self.weight = torch.nn.Parameter(self.weight.detach())
         
            self.weight_quantizer = LSQ_Quantizer(w_bit, False)
            self.weight_quantizer.init_step_size(self.weight)
    
            self.act_quantizer = LSQ_Quantizer(a_bit, True)
    
        def forward(self, x):
            quantized_weight = self.weight_quantizer(self.weight)
    
            quantized_act = self.act_quantizer(x)
    
            # quantized_act = x
    
            return F.conv2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    return LSQ_Conv2D
        





