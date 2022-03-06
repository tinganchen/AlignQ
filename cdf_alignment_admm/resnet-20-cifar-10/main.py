import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from utils.optimizer import SGD, ADMM_OPT

#from fista import FISTA
# from model import Discriminator

from data import cifar10


import warnings

warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    loader = cifar10.Data(args)

    # Create model
    print('=> Building model...')
    
    source_file = args.source_file
    
    if args.method == 'ours':
        ARCH = 'resnet'
    elif args.method in ['uniform', 'dorefa', 'llsq']:
        ARCH = 'resnet_after'      
        source_file = f'{args.method}/' + args.source_file
    elif args.method in ['apot', 'lsq']:
        ARCH = 'resnet_none'
        source_file = f'{args.method}/' + args.source_file
    else: 
        ARCH = f'resnet_{args.method}'
        source_file = f'{args.method}/' + args.source_file
    
    model_t = import_module(f'model.{ARCH}').__dict__[args.target_model](bitW = args.bitW, abitW = args.abitW, stage = args.stage).to(device)
    
    # Load pretrained weights
    if args.pretrained:
        ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        state_dict = ckpt['state_dict_t']
    
        model_dict_t = model_t.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict_t.keys()):
                model_dict_t[name] = param
        
        model_t.load_state_dict(model_dict_t)
        model_t = model_t.to(device)
        
        
        del ckpt, state_dict, model_dict_t
        
       
    models = [model_t]
    
    
    param_t = [param for name, param in model_t.named_parameters() if 'alterD' not in name and 'gamma' not in name]
    
    if 'ours' in args.method and args.bitW < 32:
        optimizer_t = SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
        param_admm = [param for name, param in model_t.named_parameters() if 'alterD' in name or 'gamma' in name]
        optimizer_admm = ADMM_OPT(param_admm)
    else:
        optimizer_t = optim.SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    
    
    scheduler_t = MultiStepLR(optimizer_t, args.lr_decay_steps, gamma = args.lr_gamma)


    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_t.load_state_dict(ckpt['state_dict_t'])

        optimizer_t.load_state_dict(ckpt['optimizer_t'])

        scheduler_t.load_state_dict(ckpt['scheduler_t'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    if 'ours' in args.method and args.bitW < 32:
        optimizers = [optimizer_t, optimizer_admm]
    else:
        optimizers = [optimizer_t]
        
    schedulers = [scheduler_t]


    block_bits = None
    
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader.loader_train, models, optimizers, epoch, block_bits)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)
        
        if args.method == 'ours' and args.bitW < 32:
            state = {
                'state_dict_t': model_t.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                
                'optimizer_admm': optimizer_admm.state_dict(),
                'optimizer_t': optimizer_t.state_dict(),
                
                #'scheduler': scheduler.state_dict(),
                'scheduler_t': scheduler_t.state_dict(),
                
                'epoch': epoch + 1
            }
        else:
            state = {
                'state_dict_t': model_t.state_dict(),
                'best_prec1': best_prec1,
                'best_prec5': best_prec5,
                
                #'optimizer': optimizer.state_dict(),
                'optimizer_t': optimizer_t.state_dict(),
                
                #'scheduler': scheduler.state_dict(),
                'scheduler_t': scheduler_t.state_dict(),
                
                'epoch': epoch + 1
            }
        checkpoint.save_model(state, epoch + 1, is_best)
        
        #compressionInfo(model_t, epoch, init = False)
        #model_p = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, model_t.state_dict(), args.t) # args.thres
        #flops, params = get_model_complexity_info(model_p.to(device), (3, 32, 32), as_strings = False, print_per_layer_stat = False)
 
    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")


def compressionInfo(model, epoch = 0, init = False):
    def total_param(model):
        N = 0
        for name, m in model.named_modules():
            if 'conv' in name and 'quantize_fn' not in name:
                param = m.weight
                N += np.prod(param.data.cpu().numpy().shape)
                if m.bias is not None:
                    param = m.bias
                    N += np.prod(param.data.cpu().numpy().shape)
        return N
        
    def total_bit(model):
        N = 0
        convs = []
        conv_quants = []
        for name, m in model.named_modules():
            if 'conv' in name and 'quantize_fn' not in name:
                convs.append(m)
            if 'conv' in name and 'quantize_fn' in name:
                conv_quants.append(m)
        
        
        convs = convs[1:]
        
        for i in range(len(convs)):
            conv = convs[i]
            quant = conv_quants[i]
            
            param = conv.weight
            
            w_bit = quant.w_bit
                
            N += np.prod(param.data.cpu().numpy().shape) * w_bit
            if conv.bias is not None:
                param = conv.bias
                N += np.prod(param.data.cpu().numpy().shape) * w_bit
        return N
    
    TP = total_param(model)
    TB = total_bit(model)
    Comp = (TP*32)/TB
    
    if init or (not args.mp):
        print('Compression rate [%d / %d]:  %.2f X' % (TP*32, TB, Comp))
        
        if not os.path.isdir(args.job_dir + '/run/plot'):
            os.makedirs(args.job_dir + '/run/plot')        
            with open(args.job_dir + 'run/plot/compressInfo_r.txt', 'w') as f:
                f.write('before, after, comp_rate \n')
        
        with open(args.job_dir + 'run/plot/compressInfo.txt', 'a') as f:
            f.write(f'{TP*32}, {TB}, {Comp}\n')
            
    else:  
        print('Final compression rate [%d / %d]:  %.2f X\n' % (TP*32, TB, Comp))
        
        if not os.path.isdir(args.job_dir + '/run/plot'):
            os.makedirs(args.job_dir + '/run/plot')        
            with open(args.job_dir + 'run/plot/compressInfo_r.txt', 'w') as f:
                f.write('epoch, before, after, comp_rate \n')
        
        with open(args.job_dir + 'run/plot/compressInfo.txt', 'a') as f:
            f.write(f'{epoch}, {TP*32}, {TB}, {Comp}\n')
        

       
def train(args, loader_train, models, optimizers, epoch, block_bits = None):
    losses_t = utils.AverageMeter()
    losses_trans = utils.AverageMeter()
    #losses_aq = utils.AverageMeter()
    #losses_kd = utils.AverageMeter()
    #losses_q = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    #model_s = models[0]
    model_t = models[0]
     
    param_t = []
    param_admm = []
    
    for name, param in model_t.named_parameters():
        if 'alterD' not in name and 'gamma' not in name:
            param_t.append((name, param))
        else:
            param_admm.append((name, param))
            
            
    cross_entropy = nn.CrossEntropyLoss()
    
    #optimizer = optimizers[0]
    optimizer_t = optimizers[0]
    
    if args.bitW < 32 and 'ours' in args.method:
        optimizer_admm = optimizers[1]
    
    # switch to train mode
    #model_s.train()
    model_t.train()
        
    num_iterations = len(loader_train)
    

    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
        '''
        p = float(num_iters) / args.num_epochs / num_iterations

        lr = adjust_learning_rate(optimizer_t, p)
        '''
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer_t.zero_grad()
        
        if args.bitW < 32 and 'ours' in args.method:
            optimizer_admm.zero_grad()
            
        ## train weights
        output_t, trans_loss = model_t(inputs)

        
        error_t = cross_entropy(output_t, targets) + trans_loss
 
        
        error_t.backward() # retain_graph = True
        
        
        losses_t.update(error_t.item(), inputs.size(0))
        
        writer_train.add_scalar('Performance_loss', error_t.item(), num_iters)
        
        
        losses_trans.update(trans_loss.item(), inputs.size(0))
        
        writer_train.add_scalar('Transformation_loss', trans_loss.item(), num_iters)


        if args.bitW < 32 and args.method == 'ours':
            idx = []
            for j, (name, param) in enumerate(param_t):
                if 'conv' in name and 'weight' in name:
                    idx.append(j)
            idx = idx[1:]
            
            w_cdf = []
            w_pdf = []

            for layer in model_t.layers:
                for conv in [layer.conv0, layer.conv1, layer.skip_conv]:
                    if conv is not None:
                        w_cdf.append(conv.quantize_fn.weight_cdf)
                        w_pdf.append(conv.quantize_fn.weight_pdf)
                        
                        
        if args.bitW < 32 and 'ours' in args.method:
            alterD_idx = []
            gamma_idx = []
            for j, (name, param) in enumerate(param_admm):
                if 'alterD' in name:
                    alterD_idx.append(j)
                elif 'gamma' in name:
                    gamma_idx.append(j)
            
            Ds = [model_t.admm0.D]
            alterDs = [model_t.admm0.alterD]
            gammas = [model_t.admm0.gamma]
            mus = [model_t.admm0.mu]
            rhos = [model_t.admm0.rho]

            for layer in model_t.layers:
                Ds.append(layer.admm0.D)
                Ds.append(layer.admm1.D)
                if layer.skip_conv is not None:
                    Ds.append(layer.admm_skip.D)
                
                alterDs.append(layer.admm0.alterD)
                alterDs.append(layer.admm1.alterD)
                if layer.skip_conv is not None:
                    alterDs.append(layer.admm_skip.alterD)
                
                gammas.append(layer.admm0.gamma)
                gammas.append(layer.admm1.gamma)
                if layer.skip_conv is not None:
                    gammas.append(layer.admm_skip.gamma)
                
                mus.append(layer.admm0.mu)
                mus.append(layer.admm1.mu)
                if layer.skip_conv is not None:
                    mus.append(layer.admm_skip.mu)
                
                rhos.append(layer.admm0.rho)
                rhos.append(layer.admm1.rho)
                if layer.skip_conv is not None:
                    rhos.append(layer.admm_skip.rho)
                        
       
        if args.bitW < 32 and args.method == 'ours':
            optimizer_t.step(idx, w_cdf, w_pdf, args.lam, args.lam2)
            optimizer_admm.step(alterD_idx, gamma_idx, Ds, alterDs, gammas, mus, rhos)
        elif 'ours' in args.method:
            optimizer_admm.step(alterD_idx, gamma_idx, Ds, alterDs, gammas, mus, rhos)
        else:
            optimizer_t.step()
        #optimizer_admm.step()
        
        ## evaluate
        prec1, prec5 = utils.accuracy(output_t, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Transform_loss: {transform_loss.val:.4f} ({transform_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    transform_loss = losses_trans,
                    top1 = top1, 
                    top5 = top5))
                
        
 
            
def test(args, loader_test, model_t, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_t.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model_t(inputs.to(device))
            loss = cross_entropy(logits, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
        
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
        
    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg
    

if __name__ == '__main__':
    main()

