import torch.nn as nn
import torch.nn.functional as F

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class ADMM(Module):
    """
    ADMM loss
    """

    def __init__(self, dim):
        super(ADMM, self).__init__()
        self.mu = 0.2
        self.rho = 0.3
        self.alterD = Parameter(torch.rand(dim, dim))
        self.gamma = Parameter(torch.rand(dim, dim))

    def forward(self, D):
        self.D = D
        alterD = self.alterD[:self.D.shape[0], :self.D.shape[1]]
        gamma = self.gamma[:self.D.shape[0], :self.D.shape[1]]
        loss_reg = self.mu * torch.mean(torch.abs(alterD)) 

        loss_constraint = self.rho / 2 * torch.mean((self.D - alterD)**2)**0.5
        loss_relax_const = torch.mean(gamma * torch.abs(self.D - alterD))
        loss = loss_reg + loss_constraint + loss_relax_const
        return loss

               
               
