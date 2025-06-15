# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import MinkowskiEngine as ME
from models.models_3d.MinkowskiEngine.examples.pointnet import (
    PointNet,
    MinkowskiPointNet,
    CoordinateTransformation,
    ModelNet40H5,
    stack_collate_fn,
    minkowski_collate_fn,
) 

# from models.models_3d.MinkowskiEngine.examples.common import seed_all
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, D=3):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=stride, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, stride=1, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MinkowskiResNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel=3, out_channel=1024, channels=(32, 64, 128, 256), D=3):
        super().__init__(D)
        self.D = D
        self.stem = nn.Sequential(
            ME.MinkowskiLinear(in_channel, channels[0]),
            ME.MinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(),
        )

        self.layer1 = BasicBlock(channels[0], channels[0], stride=1, D=D)
        self.layer2 = BasicBlock(channels[0], channels[1], stride=2, D=D)
        self.layer3 = BasicBlock(channels[1], channels[2], stride=2, D=D)
        self.layer4 = BasicBlock(channels[2], channels[3], stride=2, D=D)

        self.final = nn.Sequential(
            ME.MinkowskiLinear(channels[3], 512),
            ME.MinkowskiDropout(),
            ME.MinkowskiLinear(512, out_channel),
        )

        self.global_pool = ME.MinkowskiGlobalAvgPooling()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = self.final(x)

        return x.F  # Final feature vector
    
class GlobalMaxAvgPool(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        x = self.global_max_pool(tensor)
        y = self.global_avg_pool(tensor)
        return ME.cat(x, y)

