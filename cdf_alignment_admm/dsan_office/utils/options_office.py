import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


INDEX = 'ours_4_4_tsne'

METHOD = 'ours'
SRC = 'amazon'
TGT = 'webcam'

'''
tmux, index

'''

SRC_ONLY = False
ACT_RANGE = 2 # 3
ALPHA = 0 # step fn. slope modification
LAMBDA = 1 # sigmoid 
LAMBDA2 = 4 # sigmoid scale

PRETRAINED = True
STAGE = 'align'


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
#parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Dataset to train')

parser.add_argument('--src_data', type = str, default = f'{SRC}', help = 'The directory where the input data is stored.')
parser.add_argument('--tgt_data', type = str, default = f'{TGT}', help = 'The directory where the input data is stored.')
parser.add_argument('--src_split', action = 'store_true', default = False, help = 'Split train-test sets or not')
parser.add_argument('--tgt_split', action = 'store_true', default = False, help = 'Split train-test sets or not')

parser.add_argument('--train_split', action = 'store_true', default = False, help = 'Split train-test sets or not')

parser.add_argument('--method', type = str, default = f'{METHOD}', help = 'The directory where the input data is stored.')

parser.add_argument('--src_data_dir', type = str, default = os.getcwd() + f'/data/office31_{SRC}/', help = 'The directory where the input data is stored.')
parser.add_argument('--tgt_data_dir', type = str, default = os.getcwd() + f'/data/office31_{TGT}', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'experiment/{METHOD}/{SRC}_{TGT}/t_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_thres_{THRES}_t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'

parser.add_argument('--pretrained', type = str, default = PRETRAINED, help = 'Load pretrained model')
parser.add_argument('--stage', type = str, default = STAGE, help = 'Load pruned model')

parser.add_argument('--source_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = f'{SRC}_{TGT}/{SRC[0] + TGT[0]}_32bit/model_best.pt', help = 'The file the teacher model weights saved as.')
#parser.add_argument('--source_dir', type = str, default = 'experiment/', help = 'The directory where the teacher model saved.')
#parser.add_argument('--source_file', type = str, default = f'office/{SRC}_{TGT}/t_32_32_0/checkpoint/model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--bottle_neck', action = 'store_true', default = True, help = 'more classifier layers')
parser.add_argument('--src_only_flag', action = 'store_true', default = SRC_ONLY, help = 'Load pruned model')
parser.add_argument('--bitW', type = int, default = 4, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = 4, help = 'Quantized bitwidth.') # None

parser.add_argument('--arch', type = str, default = 'resnet', help = 'Architecture of teacher and student')
parser.add_argument('--model', type = str, default = 'resnet50_dsan', help = 'The target model.')

parser.add_argument('--num_epochs', type = int, default = 200, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 32, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 32, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.01) # 0.001
# 1. train: default = 0.01
# 2. fine_tuned: default = 5e-2
parser.add_argument('--lr_gamma', type = float, default = 0.1)

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [30, 60, 80])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'The weight decay of loss.')

parser.add_argument('--act_range', type = int, default = ACT_RANGE)

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
parser.add_argument('--alpha', type = float, default = ALPHA, help = 'Modify the approximated slope of step function.')
parser.add_argument('--param', type = float, default = 0.3, help = 'Scale the sigmoid function.')
parser.add_argument('--lam', type = float, default = LAMBDA, help = 'Scale the sigmoid function.')
parser.add_argument('--lam2', type = float, default = LAMBDA2, help = 'Scale the sigmoid function.')

## Status
parser.add_argument('--print_freq', type = int, default = 10, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

