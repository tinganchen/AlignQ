import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = '1_1_pre32_1'

'''
tmux, index

'''

METHOD = 'ours'

SRC = 'mnist'
TGT = 'mnistm'

SRC_ONLY = False
ACT_RANGE = 2 # 3
ALPHA = 10 # step fn. slope modification
#LAMBDA2 = 4 # sigmoid scale

PRETRAINED = True
STAGE = 'org'

# SEED = 12345 
# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset_root', type = str, default = f'data/{SRC}', help = 'Dataset to train')

parser.add_argument('--method', type = str, default = f'{METHOD}', help = 'The quantization method.')
parser.add_argument('--src_data', type = str, default = f'{SRC}', help = 'The directory where the input data is stored.')
parser.add_argument('--tgt_data', type = str, default = f'{TGT}', help = 'The directory where the input data is stored.')
parser.add_argument('--src_split', action = 'store_true', default = True, help = 'Split train-test sets or not')
parser.add_argument('--tgt_split', action = 'store_true', default = True, help = 'Split train-test sets or not')

#parser.add_argument('--src_data_dir', type = str, default = os.getcwd() + '/data/office31_webcam/', help = 'The directory where the input data is stored.')
#parser.add_argument('--tgt_data_dir', type = str, default = os.getcwd() + '/data/office31_amazon/', help = 'The directory where the input data is stored.')
parser.add_argument('--src_data_dir', type = str, default = os.getcwd() + f'/data/{SRC}/', help = 'The directory where the input data is stored.')
parser.add_argument('--tgt_data_dir', type = str, default = os.getcwd() + f'/data/{TGT}/', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'experiment/{METHOD}/{SRC}_{TGT}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_thres_{THRES}_t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'

parser.add_argument('--pretrained', action = 'store_true', default = PRETRAINED, help = 'Load pretrained model')
parser.add_argument('--stage', type = str, default = STAGE, help = 'Load pruned model')


parser.add_argument('--source_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = f'{SRC}_{TGT}_32bit/model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--src_only_flag', action = 'store_true', default = SRC_ONLY, help = 'Load pruned model')
parser.add_argument('--bitW', type = int, default = 1, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = 1, help = 'Quantized bitwidth.') # None

parser.add_argument('--arch', type = str, default = 'dann', help = 'Architecture of teacher and student')
parser.add_argument('--model', type = str, default = 'MNISTmodel_quant', help = 'The target model.')

parser.add_argument('--num_epochs', type = int, default = 100, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 64, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 100, help = 'Batch size for validation.')
parser.add_argument('--img_size', type = int, default = 28, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0., help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.0002)
# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.1)

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [80, 120, 180, 225])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 0., help = 'The weight decay of loss.')

parser.add_argument('--act_range', type = int, default = ACT_RANGE)

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
parser.add_argument('--alpha', type = float, default = ALPHA, help = 'Modify the approximated slope of step function.')
#parser.add_argument('--lam2', type = float, default = LAMBDA2, help = 'Scale the sigmoid function.')

## Status
parser.add_argument('--print_freq', type = int, default = 600, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

