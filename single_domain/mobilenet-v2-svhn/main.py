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


from data import svhn

from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from utils.optimizer import SGD


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
    print('=> Loading preprocessed data..')
    
    loader = svhn.Data(args)
    
    loader_train = loader.loader_train
    loader_test = loader.loader_test
    
    # Create model
    print('=> Building model...')
    
    
    
    if args.method == 'ours':
        ARCH = 'mobilenetV2'
    elif args.method in ['uniform', 'dorefa', 'llsq']:
        ARCH = 'mobilenetV2_after'      
    elif args.method in ['apot', 'lsq']:
        ARCH = 'mobilenetV2_none'
    else: 
        print(f'Method: {args.method} is not in list.')
        
    model_t = import_module(f'model.{ARCH}').__dict__[args.target_model](wbit = args.bitW, abit = args.abitW, stage = args.stage).to(device)
    
    if args.pretrained:
        # Load pretrained weights
 
        ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]
        
        new_state_dict = model_t.state_dict()
        for k, v in model_t.state_dict().items():
            if k in state_dict:
                new_state_dict[k] = v
        model_t.load_state_dict(new_state_dict)
    
    print('Finish loading state_dict.')

    param_t = [param for name, param in model_t.named_parameters()]

    if args.bitW < 32 and args.method == 'ours':
        optimizer_t = SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
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

    
    print('Start training...')

    
    for epoch in range(start_epoch, args.num_epochs):
        scheduler_t.step(epoch)

        train(args, loader.loader_train, model_t, optimizer_t, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)
        
        state = {
            'state_dict_s': model_t.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            
            'optimizer_t': optimizer_t.state_dict(),
            'scheduler_t': scheduler_t.state_dict(),
 
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")
 

       
def train(args, loader_train, model, optimizer, epoch):
    losses_t = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = model
    
    param_t = []
    
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        if 'alpha' not in name:
            param_t.append((name, param))
            
    optimizer_t = optimizer
    cross_entropy = nn.CrossEntropyLoss()
    
    # switch to train mode
    model_t.train()
        
    num_iterations = len(loader_train)
    
    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer_t.zero_grad()

        ## train weights
        output_t = model_t(inputs).to(device)

        error_t = cross_entropy(output_t, targets)

        error_t.backward() # retain_graph = True
        
        losses_t.update(error_t.item(), inputs.size(0))
        
        writer_train.add_scalar('Performance_loss', error_t.item(), num_iters)
        
        
        ## step forward
        if args.bitW < 32 and args.method == 'ours':
            idx = []
            for j, (name, param) in enumerate(param_t):
                if ('conv' in name and 'weight' in name) or ('shortcut.0' in name and 'weight' in name):
                    #print(name)
                    idx.append(j)
            #idx = idx[1:]
            
            w_cdf = []
            w_pdf = []
            
            for conv in [model_t.conv1]:
                w_cdf.append(conv.quantize_fn.weight_cdf)
                w_pdf.append(conv.quantize_fn.weight_pdf)

            for layer in model_t.layers:
                for id, conv in enumerate([layer.conv1, layer.conv2, layer.conv3, layer.shortcut]):
                    if conv is not None:
                        if id == 3:
                            conv = conv[0]
                        w_cdf.append(conv.quantize_fn.weight_cdf)
                        w_pdf.append(conv.quantize_fn.weight_pdf)
  
            for conv in [model_t.conv2]:
                w_cdf.append(conv.quantize_fn.weight_cdf)
                w_pdf.append(conv.quantize_fn.weight_pdf)
                
        if args.bitW < 32 and args.method == 'ours':
            optimizer_t.step(idx, w_cdf, w_pdf, args.lam, args.lam2)
        else:
            optimizer_t.step()

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
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses_t, 
                top1 = top1, top5 = top5))


def test(args, loader_test, model_s, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs).to(device)
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

