# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
# from mmseg.ops.wrappers import swin_Expand
from einops import rearrange
import attr
import cv2
import matplotlib.pyplot as plt

# def swin_expand(dim,dim_scale,norm_layer):
#     return swin_Expand(dim,dim_scale,norm_layer)

class swin_Expand(nn.Module):      #new!!!!
    def __init__(self, dim=128, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1)    #B,H*W,C
        x = self.expand(x)  #B,H*W,2C
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)   #B,2H,2W,C/2
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x.view(B,2*H,2*W,C//4).permute(0,3,1,2)

class swin_Expand_x4(nn.Module):      #new!!!!
    def __init__(self, dim=128, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale**2 * dim, bias=False)
        self.output_dim=dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1)    #B,H*W,C
        x = self.expand(x)  #B,H*W,2C
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2))   #B,2H,2W,C/2
        x = x.view(B, -1, C // (self.dim_scale**2))
        x = self.norm(x)

        return x.view(B,self.dim_scale*H,self.dim_scale*W,C//(self.dim_scale**2)).permute(0,3,1,2)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), final_act=True, activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.final_act = final_act
        self.out_dim = hidden_dims[0]   #new
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for affine in self.affine_layers:
            x = affine(x)
            if affine != self.affine_layers[-1] or self.final_act:
                x = self.activation(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


class up(nn.Module):
    def forward(self, x1):
        out = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=True)
        return out

# @HEADS.register_module()
class MOEHead(BaseDecodeHead):
    """
    """
    def __init__(self, feature_strides, prescale_mlp_dims=None, prescale_mlp_final_act=True,
                 afterscale_mlp_dims=[512, 256], afterscale_mlp_final_act=True, moe_mlp_dims=[512, 256], moe_conv_dims=None, activation='relu', use_linear_fuse=True,dim=256, **kwargs):
        super(MOEHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.prescale_mlp_dims = prescale_mlp_dims
        self.afterscale_mlp_dims = afterscale_mlp_dims
        self.use_linear_fuse = use_linear_fuse
        self.swin_expand = nn.ModuleList()
        self.dim=dim
        for i in range(4):
            self.swin_expand.append(swin_Expand(dim=int(256*2**i), dim_scale=2, norm_layer=nn.LayerNorm))
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        # cur_dim = sum(self.in_channels)
        # if prescale_mlp_dims is not None:
        #     self.prescale_mlp = nn.ModuleList()
        #     for in_channel in self.in_channels:
        #         mlp = MLP(in_channel, prescale_mlp_dims, prescale_mlp_final_act, activation)
        #         self.prescale_mlp.append(mlp)

        if prescale_mlp_dims is not None:    #new!!!!!
            self.prescale_mlp = nn.ModuleList()
            for i in range(len(self.in_channels)):
                # mul=2**i
                if i==0:
                    prescale_mlp_dims=(prescale_mlp_dims[-1],prescale_mlp_dims[-1])
                    mlp = MLP(self.in_channels[i], prescale_mlp_dims, prescale_mlp_final_act, activation)
                if i!=0:
                    prescale_mlp_dims=(2*prescale_mlp_dims[-1],2*prescale_mlp_dims[-1])
                    mlp =MLP(self.in_channels[i],prescale_mlp_dims,prescale_mlp_final_act,activation)
                self.prescale_mlp.append(mlp)

        cur_dim = len(self.in_channels) * prescale_mlp_dims[-1]//8

        if moe_conv_dims is not None:
            self.moe_conv = nn.ModuleList()
            conv_dims = moe_conv_dims + [len(self.in_channels)]
            for conv_dim in conv_dims:
                conv_layer = ConvModule(
                    in_channels=cur_dim,
                    out_channels=conv_dim,
                    kernel_size=3, stride=1, padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True)
                )
                cur_dim = conv_dim
                self.moe_conv.append(conv_layer)
        else:
            self.moe_conv = None


        if moe_mlp_dims is not None:
            self.moe_mlp = MLP(cur_dim, moe_mlp_dims + [len(self.in_channels)], False, activation)
        else:
            self.moe_mlp = None

        if afterscale_mlp_dims is not None:
            self.afterscale_mlp = MLP(prescale_mlp_dims[-1]//8, afterscale_mlp_dims, afterscale_mlp_final_act, activation)
        cur_dim = afterscale_mlp_dims[-1]

        if use_linear_fuse:
            self.linear_fuse = ConvModule(
                in_channels=cur_dim,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            cur_dim = embedding_dim

        self.up_x4 = swin_Expand_x4()
        # self.up =up()

        self.linear_pred = nn.Conv2d(cur_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, img_metas=None):
        x = self._transform_inputs(inputs)
        largest_size = x[0].shape[-2:]

        x_scaled = []
        for i, x_i in enumerate(x):
            if self.prescale_mlp_dims is not None:
                x_i = self.prescale_mlp[i](x_i)     #mlp

            if x_i.shape[-2:] != largest_size:  #swim uppsample
                x_i_scaled = self.swin_expand[i](x_i)
                if x_i_scaled.shape[-2:] != largest_size:
                    x_i_scaled = self.swin_expand[i-1](x_i_scaled)
                    if x_i_scaled.shape[-2:] != largest_size:
                        x_i_scaled = self.swin_expand[i-2](x_i_scaled)
            else:
                x_i_scaled = x_i
            x_scaled.append(x_i_scaled)
        x_stacked = torch.stack(x_scaled, dim=1)
        x = torch.cat(x_scaled, dim=1)

        if self.moe_conv is not None:
            for conv_layer in self.moe_conv:
                x = conv_layer(x)

        if self.moe_mlp is not None:
            x = self.moe_mlp(x)

        moe_weights = torch.softmax(x, dim=1)
        x = (x_stacked * moe_weights.unsqueeze(2)).sum(1)


        if self.afterscale_mlp_dims is not None:
            x = self.afterscale_mlp(x)

        if self.use_linear_fuse:
            x = self.linear_fuse(x)

        x = self.dropout(x)

        # x =self.up_x4(x)   #swin中将56转为224
        # x =self.up(x)   #swin中将56转为224

        x = self.linear_pred(x)

        # if img_metas is not None:
        #     case = img_metas[0]['filename'].split('/')[-1].split('.')[0]
        #     save_dir = 'results/moe_weights_cmap'
        #     weights = moe_weights.cpu().numpy()
        #     for i in range(moe_weights.shape[1]):
        #         w = weights[0,i,:,:]
        #         filename = f'{save_dir}/{case}_{i}.png'
        #         plt.imsave(filename, w, cmap='OrRd', vmin=0, vmax=0.6)
        #         # cv2.imwrite(filename, w*255)
        return x














