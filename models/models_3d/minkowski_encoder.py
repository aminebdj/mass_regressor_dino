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
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out=None, num_heads=4, dropout=0.1):
        """
        Cross Attention Fusion Module

        Args:
            dim_q: Dimension of query features (e.g., from Modality A)
            dim_kv: Dimension of key/value features (e.g., from Modality B)
            dim_out: Output feature dimension (defaults to dim_q)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dim_out = dim_out or dim_q
        self.num_heads = num_heads

        self.query_proj = nn.Linear(dim_q, self.dim_out)
        self.key_proj = nn.Linear(dim_kv, self.dim_out)
        self.value_proj = nn.Linear(dim_kv, self.dim_out)

        self.attn = nn.MultiheadAttention(embed_dim=self.dim_out, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(self.dim_out, self.dim_out)
        self.norm = nn.LayerNorm(self.dim_out)

    def forward(self, query_feats, context_feats, context_mask=None):
        """
        Forward pass of cross attention.

        Args:
            query_feats: [B, Nq, Dq] - Query features (to be updated)
            context_feats: [B, Nk, Dk] - Context/key-value features (to attend to)
            context_mask: [B, Nk] - Optional mask for context features

        Returns:
            Fused output: [B, Nq, D_out]
        """
        Q = self.query_proj(query_feats)
        K = self.key_proj(context_feats)
        V = self.value_proj(context_feats)

        attn_output, _ = self.attn(Q, K, V, key_padding_mask=context_mask)
        fused = self.out_proj(attn_output)
        fused = self.norm(fused + query_feats)  # Residual + LayerNorm

        return fused


class MinkowskiResNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel=3, out_channel=1024, channels=(32, 64, 128, 256), D=3):
        super().__init__(D)
        self.D = D
        self.stem = nn.Sequential(
            ME.MinkowskiLinear(in_channel, channels[0]),
            ME.MinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(),
        )
        clip_hiden_dim = 768

        self.layer1 = BasicBlock(channels[0], channels[0], stride=1, D=D)
        self.FL1 = CrossAttentionFusion(channels[0], clip_hiden_dim)
        self.layer2 = BasicBlock(channels[0], channels[1], stride=2, D=D)
        self.FL2 = CrossAttentionFusion(channels[1], clip_hiden_dim)
            
        self.layer3 = BasicBlock(channels[1], channels[2], stride=2, D=D)
        self.FL3 = CrossAttentionFusion(channels[2], clip_hiden_dim)
        
        
        self.layer4 = BasicBlock(channels[2], channels[3], stride=2, D=D)
        self.FL4 = CrossAttentionFusion(channels[3], clip_hiden_dim)
        
        

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

    def forward(self, x: ME.TensorField, multi_scale_clip_feats: list, num_images_per_pt: int):
        def fuse_features(x, fusion_layer, clip_f1):
            _, nq, f_dim =  clip_f1.shape
            fused_decomposed_features = []
            for i, f in enumerate(x.decomposed_features):
                ff  = fusion_layer(f[None, ...], clip_f1[num_images_per_pt*i:num_images_per_pt*(i+1), ...].reshape(1,num_images_per_pt*nq, f_dim).float())
                fused_decomposed_features.append(ff[0])
            x = ME.SparseTensor(
                    features=torch.cat(fused_decomposed_features, dim=0),
                    coordinate_manager=x.coordinate_manager,
                    coordinate_map_key=x.coordinate_map_key
                )
            return x
        fuse = False
        if multi_scale_clip_feats:
            num_layers = 4
            sampled_features = multi_scale_clip_feats[::len(multi_scale_clip_feats)//(num_layers)]
            sampled_features = [f.permute(1,0,2) for f in sampled_features]
            fuse = True
        x = self.stem(x)
        x = self.layer1(x)
        x = fuse_features(x, self.FL1, sampled_features[0]) if fuse else x 
        x = self.layer2(x)
        x = fuse_features(x, self.FL2, sampled_features[1]) if fuse else x
        
        x = self.layer3(x)
        x = fuse_features(x, self.FL3, sampled_features[2]) if fuse else x
        
        x = self.layer4(x)
        x = fuse_features(x, self.FL4, sampled_features[3]) if fuse else x
        

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

