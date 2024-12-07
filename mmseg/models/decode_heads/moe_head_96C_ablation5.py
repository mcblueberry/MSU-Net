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
from einops import rearrange
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
import cv2
import matplotlib.pyplot as plt

class swin_Expand(nn.Module):      #new!!!!
    def __init__(self, dim=96, dim_scale=2, norm_layer=nn.LayerNorm):
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
    def __init__(self, dim=96, dim_scale=4, norm_layer=nn.LayerNorm):
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
class Down_channel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.downch = nn.Linear(c, 192)

    def forward(self, x):
        B,C,H,W=x.shape
        x=x.view(B,C,H,W).permute(0,2,3,1)
        x = self.downch(x)
        x=x.view(B,-1,H,W)
        return x

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
        self.out_dim = hidden_dims[-1]
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

class Down_channel(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.downch=nn.Linear(in_ch,in_ch//2)
    def forward(self,x):
        B,C,H,W =x.shape
        x=x.view(B,C,-1).permute(1,0,2)
        x=x.contiguous().view(C,-1).permute(1,0)
        x=self.downch(x)
        x=x.view(B,C//2,H,W)
        return x
# @HEADS.register_module()
class MOEHead(BaseDecodeHead):
    """
    """
    def __init__(self, feature_strides, prescale_mlp_dims=None, prescale_mlp_final_act=True,
                 afterscale_mlp_dims=[512, 256], afterscale_mlp_final_act=True, moe_mlp_dims=[512, 256],
                 moe_conv_dims=None, activation='relu', use_linear_fuse=True,dim=96, in_ch=32, out_ch=64, **kwargs):
        super(MOEHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.prescale_mlp_dims = prescale_mlp_dims
        self.afterscale_mlp_dims = afterscale_mlp_dims
        self.use_linear_fuse = use_linear_fuse
        self.in_channel=[384,768,1536]
        self.down_chan=nn.ModuleList()
        self.swin_expand = nn.ModuleList()
        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ch_down1 = Down_channel(384)
        self.ch_down2 = Down_channel(768)
        self.ch_down3 = Down_channel(1536)

        for i in range(4):
            self.swin_expand.append(swin_Expand(dim=int(192*2**i), dim_scale=2, norm_layer=nn.LayerNorm))

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        cur_dim = sum(self.in_channels)
        # if prescale_mlp_dims is not None:
        #     self.prescale_mlp = nn.ModuleList()
        #     for in_channel in self.in_channels:
        #         mlp = MLP(in_channel, prescale_mlp_dims, prescale_mlp_final_act, activation)
        #         self.prescale_mlp.append(mlp)

        cur_dim = len(self.in_channels) * prescale_mlp_dims[-1]
        self.up_x4 = swin_Expand_x4()
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
            self.afterscale_mlp = MLP(prescale_mlp_dims[-1], afterscale_mlp_dims, afterscale_mlp_final_act, activation)
        cur_dim = afterscale_mlp_dims[-1]

        if use_linear_fuse:
            self.linear_fuse = ConvModule(
                in_channels=cur_dim,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='BN', requires_grad=True)
            )
            cur_dim = embedding_dim

        self.linear_pred = nn.Conv2d(cur_dim, self.num_classes, kernel_size=1)



    def forward(self, inputs, img_metas=None):
        x = self._transform_inputs(inputs)
        largest_size = x[0].shape[-2:]
        x1=x[0]
        x2=x[1]
        x3=x[2]
        x4=x[3]

        x_scaled = []
        for i, x_i in enumerate(x):

            if x_i.shape[-2:] != largest_size:
                # x_i_scaled = self.swin_expand[i](x_i)
                x1 = self.swin_expand[i](x_i)
                if x1.shape[-2:] == largest_size:
                    x1 = torch.cat([x1, x[0]], dim=1)
                    x_i_scaled = self.ch_down1(x1)
                if x_i_scaled.shape[-2:] != largest_size:
                    x2 = self.swin_expand[i - 1](x_i_scaled)
                    if x2.shape[-2:] == largest_size:
                        x2 = torch.cat([x2, x[1]], dim=1)
                        x_i_scaled = self.ch_down2(x2)
                    if x3.shape[-2:] != largest_size:
                        x3 = self.swin_expand[i - 2](x_i_scaled)
                        if x3.shape[-2:] == largest_size:
                            x3 = torch.cat([x3, x[2]], dim=1)
                            x_i_scaled = self.ch_down3(x3)

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
        x = self.up_x4(x)
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

if __name__ == '__main__':
    #[64, 128, 256, 512]
    input_1 = torch.randn((1,192,56,56,))
    input_2 = torch.randn((1, 384, 28, 28,))
    input_3 = torch.randn((1, 768, 14, 14,))
    input_4 = torch.randn((1, 1536, 7, 7,))
    print(input_1.shape)
    print(input_2.shape)
    print(input_3.shape)
    print(input_4.shape)
    decode = MOEHead(prescale_mlp_dims=[192, 192],  # new
                          prescale_mlp_final_act=None,
                          afterscale_mlp_dims=[96, 96],  # new
                          afterscale_mlp_final_act=True,
                          moe_mlp_dims=None,  # new
                          moe_conv_dims=[96],
                          in_channels=[192, 384, 768, 1536],
                          in_index=[0, 1, 2, 3],
                          feature_strides=[4, 8, 16, 32],
                          channels=96,
                          dropout_ratio=0.1,
                          num_classes=4,
                          align_corners=False,
                          decoder_params=dict(embed_dim=96),
                          # dim=96,
                          loss_decode=dict(type='StructurelLoss'))
    out = decode([input_1,input_2,input_3,input_4])
    print(out.shape)











