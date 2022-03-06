import torch
from torch.optim.optimizer import Optimizer, required
from utils.options import args
import numpy as np

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def sigmoid_d(x, lam):
    return sigmoid(x) * (1-sigmoid(x)) * lam

def transform(w, lam2):
    return (( (w + 0.5) * (2 ** args.bitW - 1) ) % 1) * lam2 * 2

class ADMM_OPT(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    This is a modified version by Chen Shangyu.
    Basically, it don't add on gradient to the variable, instead it stores updated gradient (by lr and decay)
    back to gradient for further used.
    """

    def __init__(self, params):
        defaults = dict()
        
        super(ADMM_OPT, self).__init__(params, defaults)
        
        

    def step(self, alterD_idx, gamma_idx, Ds, alterDs, gammas, mus, rhos, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #weight_decay = group['weight_decay']
            #momentum = group['momentum']
            #dampening = group['dampening']
            #nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                '''
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                '''
                if args.bitW < 32:
                    if i in alterD_idx:
                        j = alterD_idx.index(i)
                        D = Ds[j]
                        gamma = gammas[j]
                        rho = rhos[j]
                        mu = mus[j]
                        
                        D_ = torch.zeros_like(gamma)
                        D_[:D.shape[0], :D.shape[1]] = D
                        
                        V = D_ + 1/rho * gamma 
                        
                        if torch.norm(V, 2) > (mu/rho):
                            alterD = (1 - mu/rho / torch.norm(V, 2)) * V
                        else:
                            alterD = torch.zeros_like(p.data)
                        
                        p.data = alterD
                        
                    elif i in gamma_idx:
                        j = gamma_idx.index(i)
                        D = Ds[j]
                        rho = rhos[j]
                        
                        #D_ = torch.zeros_like(gamma)
                        #D_[:D.shape[0], :D.shape[1]] = D
                        
                        p.data = p.data + rho * (D_ - alterD)
                        
                    else:
                        p.grad.data = d_p
                        p.data.add_(-group['lr'], d_p)
                        #
                    
                else:
                    p.data.add_(-group['lr'], d_p)
                    p.grad.data = d_p

        return loss
    
    
class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    This is a modified version by Chen Shangyu.
    Basically, it don't add on gradient to the variable, instead it stores updated gradient (by lr and decay)
    back to gradient for further used.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, idx, w_cdf, w_pdf, lam, lam2, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                if args.bitW < 32:
                    if i in idx:
                        weight_cdf = w_cdf[idx.index(i)].data
                        approx = sigmoid_d(transform(weight_cdf, lam2), lam)
                        
                        weight_pdf = w_pdf[idx.index(i)].data


                        # gradient clipping
                        '''
                        I = torch.abs(d_p* approx * weight_pdf)# 
                        
                        thres = torch.mean(I) + 3 * torch.std(I) #np.percentile(I.cpu(), 95)
                        
                        new_g = (I < thres) * d_p * approx * weight_pdf  + (I >= thres) * thres * torch.sign(d_p * approx * weight_pdf)
                        '''
 
                        # gradient approximation
                        p.grad.data = d_p * approx * weight_pdf
                        #p.data.add_(-group['lr'], d_p)
                        p.data.add_(-group['lr'], d_p)
                        
                    else:
                        p.grad.data = d_p
                        p.data.add_(-group['lr'], d_p)
                        #
                    
                else:
                    p.data.add_(-group['lr'], d_p)
                    p.grad.data = d_p

        return loss


