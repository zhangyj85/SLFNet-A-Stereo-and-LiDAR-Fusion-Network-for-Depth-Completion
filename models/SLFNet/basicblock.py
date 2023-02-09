from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def conv_bn_relu(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=True):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=dilation if dilation > 1 else padding, dilation=dilation, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

def convt_bn_relu(in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class ResBlock(nn.Module):
    # 基本残差块：x -> conv2d + bn + ReLU -> conv2d + bn -> y; output = x + y.
    expansion = 1                                                               # 默认扩张卷积的扩张因子 = 1，即不扩张；该变量用于特征提取模块的_make_layer()
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, padding=0, dilation=1):
        super(ResBlock, self).__init__()                                      # 继承父类nn.Module的__init__属性
        self.conv1 = conv_bn_relu(in_planes, out_planes, 3, stride, padding, dilation, bn=True, relu=True)
        self.conv2 = conv_bn_relu(out_planes, out_planes, 3, 1, 1, 1, bn=True, relu=False)
        self.downsample = downsample                                            # downsample是一个网络层（conv2d + bn），对输入进行降维操作以便进行 short cat； downsample != 降采样
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):                                                       # x -> conv2d + BN + ReLU -> conv2d + BN
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)      # downsample是一个网络层（conv2d + bn），对输入进行降采样并增加通道数

        out += x                        # 对输入进行跳跃连接

        return self.lrelu(out)
