"""DANN model."""
import os
import torch
import torch.nn as nn
from .functions import ReverseLayerF
from torchvision import models

from .quantization import *

#from .quantization_lsq import * # LSQ
#from .quantization_llsq import * # LLSQ
#from .quantization_apot import * # APOT
#from .quantization_dorefa import * # dorefa
#from .quantization_uniform import * # uniform

#from .quantization_bwn import *
#from .quantization_bwnf import *

import torch.utils.model_zoo as model_zoo
from utils.options import args

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = m.weight.size()
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
        


class Classifier(nn.Module):
    """ SVHN architecture without discriminator"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.restored = False

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 1, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        class_output = self.class_classifier(feature)

        return class_output


class MNISTmodel(nn.Module):
    """ MNIST architecture
    +Dropout2d, 84% ~ 73%
    -Dropout2d, 50% ~ 73%
    """

    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output

class MNISTmodel_quant(nn.Module):
    """ MNIST architecture
    +Dropout2d, 84% ~ 73%
    -Dropout2d, 50% ~ 73%
    """

    def __init__(self, stage, wbit, abit):
        super(MNISTmodel_quant, self).__init__()
        self.restored = False
        
        if args.method == 'lsq':
            Conv2d = conv2d_Q_fn(w_bit = wbit, a_bit = abit, stage = stage)
        else:
            Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
        
        if args.method in ['ours']:
            '''ours'''
            self.feature = nn.Sequential(
                Conv2d(in_channels=3, out_channels=32,
                       kernel_size=(5, 5)),  # 3 28 28, 32 24 24
                nn.BatchNorm2d(32),
                activation_quantize_fn(a_bit = abit, stage = stage),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
                Conv2d(in_channels=32, out_channels=48,
                       kernel_size=(5, 5)),  # 48 8 8
                nn.BatchNorm2d(48),
                activation_quantize_fn(a_bit = abit, stage = stage),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
            )
        elif args.method in ['dorefa', 'uniform', 'llsq', 'bwn', 'bwnf']:
            '''dorefa, uniform, llsq, bwn, bwnf'''
            self.feature = nn.Sequential(
                Conv2d(in_channels=3, out_channels=32,
                       kernel_size=(5, 5)),  # 3 28 28, 32 24 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                activation_quantize_fn(a_bit = abit, stage = stage),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
                Conv2d(in_channels=32, out_channels=48,
                       kernel_size=(5, 5)),  # 48 8 8
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                activation_quantize_fn(a_bit = abit, stage = stage),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
            )
        elif args.method in ['apot', 'lsq']:
            '''APOT, LSQ, the activation quantization is done in self-defined Conv'''
            self.feature = nn.Sequential(
                Conv2d(in_channels=3, out_channels=32,
                       kernel_size=(5, 5)),  # 3 28 28, 32 24 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
                Conv2d(in_channels=32, out_channels=48,
                       kernel_size=(5, 5)),  # 48 8 8
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
            )
        else:
            '''ORG'''
            self.feature = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32,
                       kernel_size=(5, 5)),  # 3 28 28, 32 24 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
                nn.Conv2d(in_channels=32, out_channels=48,
                       kernel_size=(5, 5)),  # 48 8 8
                nn.BatchNorm2d(48),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
            )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, args.img_size, args.img_size) # input_data.data.shape[0], 3, 32, 32
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output
    
 
    
