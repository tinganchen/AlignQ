import os
import numpy as np

#from utils.options import args
from utils.options_office import args
import utils.common as utils
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from utils.optimizer import SGD, ADMM_OPT


from data import office, split

#from ptflops import get_model_complexity_info # from thop import profile

import warnings, math

warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    
    tgt_best_prec1 = 0.0
    tgt_best_prec5 = 0.0
    tgt_best_prec1_domain = 0.0

    src_best_prec1 = 0.0
    src_best_prec5 = 0.0
    src_best_prec1_domain = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # split train and test
    if args.train_split:
        split.office31(train_ratio = 0.8).forward(dir_path = 'office31_split')
    
    # data loader
    
    src_loader = office.Data(args, 'data', f'{args.src_data}')
    tgt_loader = office.Data(args, 'data', f'{args.tgt_data}')
    
    src_data_loader = src_loader.loader_train
    src_data_loader_eval = src_loader.loader_test
    
    tgt_data_loader = tgt_loader.loader_train
    tgt_data_loader_eval = tgt_loader.loader_test
    

    # Create model
    print('=> Building model...')

    # load training model
    if args.method == 'ours':
        ARCH = 'resnet'
    else: 
        print('Method can only be AlignQ (ours).')
        
    model_t = import_module(f'model.{ARCH}').__dict__[args.model](wbit = args.bitW, abit = args.abitW, stage = args.stage).to(device)
    
    #compressionInfo(model_t, epoch = 0, init = True)

    # Load pretrained weights
    if args.pretrained:
        src_file = 'ours/' + args.source_file
        ckpt = torch.load(args.source_dir + src_file, map_location = device)
        state_dict = ckpt['state_dict_t']
    
        model_dict_t = model_t.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict_t.keys()):
                model_dict_t[name] = param
        
        model_t.load_state_dict(model_dict_t)
        model_t = model_t.to(device)

        del ckpt, state_dict, model_dict_t
        
       
    models = [model_t]

    
    param_t = [param for name, param in model_t.named_parameters()]
    
    if args.method == 'ours' and args.bitW < 32:
        optimizer_t = SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
        param_admm = [param for name, param in model_t.named_parameters() if 'alterD' in name or 'gamma' in name]
        optimizer_admm = ADMM_OPT(param_admm)
    else:
        optimizer_t = optim.SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    
    scheduler_t = StepLR(optimizer_t, args.lr_decay_step, gamma = args.lr_gamma)


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

    '''
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
    '''

    optimizers = [optimizer_t, optimizer_admm]
    schedulers = [scheduler_t]

    #optimizers = [optimizer, optimizer_t]
    #schedulers = [scheduler, scheduler_t]

    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, src_data_loader, tgt_data_loader, models, optimizers, epoch)
        
        tgt_test_prec1, tgt_test_prec5, tgt_test_prec1_domain = test(args, tgt_data_loader_eval, model_t, epoch, 'target')
        src_test_prec1, src_test_prec5, src_test_prec1_domain = test(args, src_data_loader_eval, model_t, epoch, 'source')
        
        
        is_best = tgt_best_prec1 < tgt_test_prec1
        tgt_best_prec1 = max(tgt_test_prec1, tgt_best_prec1)
        tgt_best_prec5 = max(tgt_test_prec5, tgt_best_prec5)
        
        tgt_best_prec1_domain = max(tgt_test_prec1_domain, tgt_best_prec1_domain)
        
        src_best_prec1 = max(src_test_prec1, src_best_prec1)
        src_best_prec5 = max(src_test_prec5, tgt_best_prec5)
        
        src_best_prec1_domain = max(src_test_prec1_domain, src_best_prec1_domain)


        state = {
            'state_dict_t': model_t.state_dict(),
            
            'tgt_best_prec1': tgt_best_prec1,
            'tgt_best_prec5': tgt_best_prec5,
            'tgt_best_prec1_domain': tgt_best_prec1_domain,
            
            'src_best_prec1': src_best_prec1,
            'src_best_prec5': src_best_prec5,
            'src_best_prec1_domain': src_best_prec1_domain,
            
            'optimizer_admm': optimizer_admm.state_dict(),
            'optimizer_t': optimizer_t.state_dict(),
            
            #'scheduler': scheduler.state_dict(),
            'scheduler_t': scheduler_t.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
        #compressionInfo(model_t, epoch, init = False)
        #model_p = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, model_t.state_dict(), args.t) # args.thres
        #flops, params = get_model_complexity_info(model_p.to(device), (3, 32, 32), as_strings = False, print_per_layer_stat = False)
 
    print_logger.info(f'Best @prec1: {tgt_best_prec1:.3f} @prec5: {tgt_best_prec5:.3f} [Target class]')
    print_logger.info(f'Best @prec1: {src_best_prec1:.3f} @prec5: {src_best_prec5:.3f} [Source class]')
 


def adjust_learning_rate(optimizer, p):
    
    lr_0 = args.lr
    alpha = args.alpha
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        '''
    # office & mnist-svhn
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr'''
    return lr


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
        
        '''
        conv = convs[0]
        param = conv.weight

        N += np.prod(param.data.cpu().numpy().shape) * 32
        if conv.bias is not None:
            param = conv.bias
            N += np.prod(param.data.cpu().numpy().shape) * 32
            '''
        
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
        
        
       
def train(args, src_data_loader, tgt_data_loader, models, optimizers, epoch):
    losses_t = utils.AverageMeter()
    losses_src_class = utils.AverageMeter()
    losses_src_domain = utils.AverageMeter()
    losses_tgt_domain = utils.AverageMeter()
    #losses_q = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    #model_s = models[0]
    model_t = models[0]
    
    param_t = []
    param_admm = []
    
    for name, param in model_t.named_parameters():
        if 'alterD' in name or 'gamma' in name:
            param_admm.append((name, param))
        param_t.append((name, param))    
            
    criterion = nn.CrossEntropyLoss()
    
    #optimizer = optimizers[0]
    optimizer_t = optimizers[0]
    
    if args.bitW < 32 and args.method == 'ours':
        optimizer_admm = optimizers[1]
        
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.num_epochs), 0.75)
    
    if args.method == 'ours':
        optimizer_t = SGD([
                {'params': model_t.feature.parameters()},
                {'params': model_t.class_classifier.parameters(), 'lr': LEARNING_RATE},
                {'params': model_t.domain_classifier.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.weight_decay)
    else:   
        optimizer_t = optim.SGD([
                    {'params': model_t.feature.parameters()},
                    {'params': model_t.class_classifier.parameters(), 'lr': LEARNING_RATE},
                    {'params': model_t.domain_classifier.parameters(), 'lr': LEARNING_RATE},
                ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.weight_decay)
       
    # switch to train mode
    #model_s.train()
    model_t.train()
        
    num_iterations = max(len(src_data_loader), len(tgt_data_loader))
    data_zip = enumerate(zip(src_data_loader, tgt_data_loader), 1)

    for i, ((images_src, class_src), (images_tgt, _)) in data_zip:
        
        num_iters = num_iterations * epoch + i
        
        p = float(num_iters) / args.num_epochs / num_iterations
        alpha = 2. / (1. + np.exp(-10 * p) +1e-6) - 1
        
        
        # prepare domain label
        size_src = len(images_src)
        size_tgt = len(images_tgt)
        label_src = torch.zeros(size_src).long().to(device)  # source 0
        label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

        # make images variable
        class_src = class_src.to(device)
        images_src = images_src.to(device)
        images_tgt = images_tgt.to(device)
                
        #inputs = inputs.to(device)
        #targets = targets.to(device)
        

        optimizer_t.zero_grad()
        
        if args.bitW < 32 and args.method == 'ours':
            optimizer_admm.zero_grad()
            
        # train on source domain
        src_class_output, src_domain_output, src_trans_loss = model_t(images_src, alpha=alpha)
        src_loss_class = criterion(src_class_output, class_src)
        src_loss_domain = criterion(src_domain_output, label_src)

        # train on target domain
        _, tgt_domain_output, tgt_trans_loss = model_t(images_tgt, alpha=alpha)
        tgt_loss_domain = criterion(tgt_domain_output, label_tgt)

        loss = src_loss_class + src_loss_domain + tgt_loss_domain + src_trans_loss + tgt_trans_loss
        if args.src_only_flag:
            loss = src_loss_class

        # optimize dann
        loss.backward()
        #optimizer_t.step()


        ## train weights        
        losses_t.update(loss.item(), size_src)
        
        losses_src_class.update(src_loss_class.item(), size_src)
        losses_src_domain.update(src_loss_domain.item(), size_src)
        losses_tgt_domain.update(tgt_loss_domain.item(), size_src)
    
    
        writer_train.add_scalar('Performance_loss', loss.item(), num_iters)
        writer_train.add_scalar('Source_class_loss', src_loss_class.item(), num_iters)
        
        if args.src_only_flag:
            writer_train.add_scalar('Source_domain_loss', src_loss_domain.item(), num_iters)
            writer_train.add_scalar('Target_domain_loss', tgt_loss_domain.item(), num_iters)
        
        if args.method == 'ours':
            if args.bitW < 32:
                idx = []
                for j, (name, param) in enumerate(param_t):
                    if ('conv' in name or 'downsample.0' in name) and 'weight' in name:
                        idx.append(j)
                idx = idx[1:]
                
                w_cdf = []
                w_pdf = []
        
                for layer in [model_t.feature.layer1, model_t.feature.layer2, model_t.feature.layer3, model_t.feature.layer4]:
                    for block in layer:
                        for k, conv in enumerate([block.conv1, block.conv2, block.conv3, block.downsample]):
                            if conv is not None:
                                if k == 3:
                                    conv = conv[0]
                                w_cdf.append(conv.quantize_fn.weight_cdf)
                                w_pdf.append(conv.quantize_fn.weight_pdf)
                        
        
        
        if args.bitW < 32 and args.method == 'ours':
            alterD_idx = []
            gamma_idx = []
            for j, (name, param) in enumerate(param_admm):
                if 'alterD' in name:
                    alterD_idx.append(j)
                elif 'gamma' in name:
                    gamma_idx.append(j)
            
            Ds = []
            alterDs = []
            gammas = []
            mus = []
            rhos = []

            for layers in [model_t.feature.layer1, model_t.feature.layer2, model_t.feature.layer3, model_t.feature.layer4]:
                for layer in layers:
                    Ds.append(layer.admm0.D)
                    
                    alterDs.append(layer.admm0.alterD)
 
                    gammas.append(layer.admm0.gamma)
                    
                    mus.append(layer.admm0.mu)
   
                    rhos.append(layer.admm0.rho)


        if args.bitW < 32 and args.method == 'ours':
            optimizer_t.step(idx, w_cdf, w_pdf, args.lam, args.lam2)
            optimizer_admm.step(alterD_idx, gamma_idx, Ds, alterDs, gammas, mus, rhos)
        else:
            optimizer_t.step()
        #optimizer_admm.step()

        
        ## evaluate
        prec1, prec5 = utils.accuracy(src_class_output, class_src, topk = (1, 5))
        top1.update(prec1[0], size_src)
        top5.update(prec5[0], size_src)
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            if args.src_only_flag:
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Source_class_loss: {src_class_loss.val:.4f} ({src_class_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    src_class_loss = losses_src_class,
                    top1 = top1, 
                    top5 = top5))
            else:
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Source_class_loss: {src_class_loss.val:.4f} ({src_class_loss.avg:.4f})\n'
                    'Source_domain_loss: {src_domain_loss.val:.4f} ({src_domain_loss.avg:.4f})\n'
                    'Target_domain_loss: {tgt_domain_loss.val:.4f} ({tgt_domain_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    src_class_loss = losses_src_class,
                    src_domain_loss = losses_src_domain,
                    tgt_domain_loss = losses_tgt_domain,
                    top1 = top1, 
                    top5 = top5))
                
      
 
def test(args, loader_test, model_t, epoch, flag):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    top1_domain = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model_t.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if flag == 'target':
                labels_domain = torch.ones(inputs.size(0)).long().to(device)
            else:
                labels_domain = torch.zeros(inputs.size(0)).long().to(device)
    
            preds, domain, _ = model_t(inputs, alpha=0)
    
            loss = criterion(preds, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
            
            # image classification results
            prec1, prec5 = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
            
            # domain classification results
            prec1 = utils.accuracy(domain, labels_domain, topk = (1, ))
            top1_domain.update(prec1[0], inputs.size(0))

            
            writer_test.add_scalar('Test-top-1_domain', top1_domain.avg, num_iters)
    
    print_logger.info(f'{flag}')
    print_logger.info(f'Prec@1 {int(top1.avg*10**3)/10**3} Prec@5 {int(top5.avg*10**3)/10**3} (Image)')
    print_logger.info(f'Prec@1 {int(top1_domain.avg*10**3)/10**3} (Domain)\n=======================================================\n')

    return top1.avg, top5.avg, top1_domain.avg
    


if __name__ == '__main__':
    main()

