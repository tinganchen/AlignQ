import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

from utils.options_office import args
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

import numpy as np

from .quantization_uniform import *
#from .quantization_dorefa import *
#from .quantization_llsq import *

device = torch.device(f"cuda:{args.gpus[0]}")


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

'''
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
            
        weight_pdf = torch.exp(normal.log_prob(tensor)) * args.act_range
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
      #self.weight_reconstruct = x
    else:
      self.weight_cdf, self.weight_pdf = cdf(torch.mean(x), torch.std(x), 'w')(x)
      #self.weight_cdf, self.weight_pdf = cdf(torch.zeros(1).to(device), torch.ones(1).to(device))(x)
      
      self.weight_q = self.uniform_q(self.weight_cdf)
      
      #self.weight_reconstruct = icdf(torch.mean(x), torch.std(x))(self.weight_q)
      
    if self.stage == 'first':
        return x
    else:
        return self.weight_q # self.weight_reconstruct


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit, stage):
    super(activation_quantize_fn, self).__init__()
    #assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.stage = stage
    
    self.uniform_q = uniform_quantize(k = a_bit)
    
    
  def forward(self, x):
    if self.a_bit == 32:
      activation_cdf = x
      activation_q = x
      #self.activation_reconstruct = x
    else:
      activation_cdf, activation_pdf = cdf(torch.zeros(1).to(device), torch.ones(1).to(device), 'a')(x)
      activation_q = self.uniform_q(activation_cdf)
     
    if self.stage == 'first':
        return x
    else:
        return activation_q # self.activation_reconstruct
    


def conv2d_Q_fn(w_bit, stage):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

      self.quantize_fn = weight_quantize_fn(w_bit = w_bit, stage = stage)
    
    def forward(self, input, order=None):
      
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit, stage):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit = w_bit, stage = stage)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
'''
def conv3x3(wbit, stage, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(wbit, stage, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, wbit, abit, stage, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(wbit, stage, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(wbit, stage, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q2 = activation_quantize_fn(a_bit = abit, stage = stage)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_q1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_q2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, wbit, abit, stage, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(wbit, stage, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(wbit, stage, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(wbit, stage, width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.act_q1 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q2 = activation_quantize_fn(a_bit = abit, stage = stage)
        self.act_q3 = activation_quantize_fn(a_bit = abit, stage = stage)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.act_q1(out)
      
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.act_q2(out)
       
        out = self.conv3(out)
        out = self.bn3(out)
        
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.act_q3(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, wbit, abit, stage, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        self.wbit = wbit
        self.abit = abit
        self.stage = stage
        
        self.act_q0 = activation_quantize_fn(a_bit = abit, stage = stage)
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        Conv2d = conv2d_Q_fn(w_bit = wbit, stage = stage)
        
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.wbit, self.stage, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.wbit, self.abit, self.stage, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.wbit, self.abit, self.stage, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)      
        x = self.relu(x)
        x = self.act_q0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        x = self.fc(feature)

        return feature

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(wbit, abit, stage, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(wbit, abit, stage, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        
        model_dict = model.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict.keys()):
                model_dict[name] = param
        
        model.load_state_dict(model_dict)

    return model


    
def resnet18_quant(wbit, abit, stage, pretrained=True, progress=True, **kwargs):
    return _resnet(wbit, abit, stage, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34_quant(wbit, abit, stage, pretrained=True, progress=True, **kwargs):
    return _resnet(wbit, abit, stage, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50_quant(wbit, abit, stage, pretrained=True, progress=True, **kwargs):
  return _resnet(wbit, abit, stage, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# DANN
class DANN(nn.Module):
    def __init__(self, arch, wbit, abit, stage, num_classes=31):
        super(DANN, self).__init__()
        self.feature = arch(wbit, abit, stage)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc3', nn.Linear(2048, num_classes))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc2', nn.Linear(2048, 2))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 2048)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def resnet18_dann(wbit, abit, stage, **kwargs):
  return DANN(resnet18_quant, wbit, abit, stage)

def resnet34_dann(wbit, abit, stage, **kwargs):
  return DANN(resnet34_quant, wbit, abit, stage)

def resnet50_dann(wbit, abit, stage, **kwargs):
  return DANN(resnet50_quant, wbit, abit, stage)

# MDD
class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, arch, wbit, abit, stage, use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = arch(wbit, abit, stage)
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(2048, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv


