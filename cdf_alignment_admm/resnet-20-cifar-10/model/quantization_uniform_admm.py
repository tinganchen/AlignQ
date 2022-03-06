import torch
import torch.nn as nn
import math

import torch.nn.functional as F

import time
import numpy as np

from utils.options import args

#from utils.gcn import GCN

#from utils.admm import ADMM

device = torch.device(f"cuda:{args.gpus[0]}")


def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class cdf(nn.Module):
    def __init__(self, m, s, quant_src):
        super(cdf, self).__init__()
    
        self.m = m
        self.s = s
        self.quant_src = quant_src

    def forward(self, tensor):
        normal = torch.distributions.Normal(self.m, self.s)
        cdf = normal.cdf(tensor)
        #weight_cdf = (cdf - torch.min(cdf)) / (torch.max(cdf) - torch.min(cdf)) * 2 - 1
        weight_cdf = cdf * 2 - 1
        
        if self.quant_src == 'a':
            weight_cdf = weight_cdf * args.act_range
            
        weight_pdf = torch.exp(normal.log_prob(tensor)) * 2
        return weight_cdf, weight_pdf
    
class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit, stage):
    super(weight_quantize_fn, self).__init__()
    #assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
        
    self.stage = stage
    
    self.uniform_q = uniform_quantize(k=self.w_bit)

  def forward(self, x):

    if self.w_bit == 32: 
      self.weight_cdf = x
      self.weight_q = x
      return x
    else:
      #self.weight_cdf, self.weight_pdf = cdf(torch.mean(x), torch.std(x), 'w')(x)

      self.weight_q = self.uniform_q(x)
      
      if self.w_bit == 32:
          return self.weight_cdf
      else:
          return self.weight_q # self.weight_reconstruct


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit, stage, admm):
    super(activation_quantize_fn, self).__init__()
    #assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.stage = stage
    
    self.uniform_q = uniform_quantize(k = a_bit)
    '''
    self.embed_layer = GCN(nfeat = ndim, nhid = int(ndim/2), nclass = int(ndim/4), dropout = 0.8)
    self.opt = ADMM()
    '''
    self.opt = admm

  def forward(self, x):
    if self.a_bit == 32 and self.stage != 'align':
        activation_cdf = x
        activation_q = x
        trans_loss = 0
        return x, trans_loss
    else:
        activation_cdf = x
        activation_q = self.uniform_q(activation_cdf)
      
        if 'ours' in args.method and self.a_bit < 32:
            ## tansformation loss
            # corr matrix before & after the transformation
            x_feat = x.view(x.shape[0], -1)
            x_trans_feat = activation_cdf.view(x.shape[0], -1)
            
            corr_mat = corr(x_feat, x_feat)
            corr_mat_trans = corr(x_trans_feat, x_trans_feat)
            
            # admm optimization to minimize the discrepancy of the two corr matrices
            D = corr_mat_trans - corr_mat
            trans_loss = self.opt(D)
        
        else:
            trans_loss = 0
              

        if self.a_bit == 32:
            return activation_cdf, trans_loss
        else:
            return activation_q, trans_loss

def corr(x, y):
    x_std = (x - torch.mean(x, dim = 0)) / torch.std(x, dim = 0)
    y_std = (y - torch.mean(y, dim = 0)) / torch.std(y, dim = 0)
    return torch.matmul(x_std, torch.transpose(y_std, 0, 1)) / x_std.shape[1]


def conv2d_Q_fn(w_bit, stage):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

      self.quantize_fn = weight_quantize_fn(w_bit = w_bit, stage = stage)
    
    def forward(self, input, order=None):
      
      weight_q = self.quantize_fn(self.weight)

      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

