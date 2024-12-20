# code is borrowed from the original repo and fit into our training framework   11.829G 28.313M 7.668G 28.230M

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging
from mmseg.models.decode_heads.moe_head_96C import *
# from mmseg.models.decode_heads.moe_head_256C import *
import math
import copy
from thop import profile,clever_format


from model.SimAM import simam_module


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class channel_shuffle(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x, groups=2):
        B,C,H,W = x.shape
        channels_per_group = C // groups
        x = x.view(B, groups, channels_per_group,H, W)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(B,-1,H,W)
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


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class PatchMerging1(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """

    def __init__(self, dim, out_dim, norm=nn.BatchNorm2d, proj_type='depthwise'):
        super().__init__()
        self.dim = dim
        if proj_type == 'linear':
            self.reduction = nn.Conv2d(4*dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(4*dim, out_dim)

        self.norm = norm(4*dim)

    def forward(self, x, semantic_map=None):
        """
        x: B, C, H, W
        """
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], 1) # B, 4C, H, W

        x = self.norm(x)
        x = self.reduction(x)

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
       windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
           f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=12, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(12, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
           f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = nn.LayerNorm


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # absolute position embedding位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.inc =DoubleConv(in_ch=3,out_ch=24)
        self.down1 =Down(12,24)
        self.down2 =Down(96,192)
        self.down3 =Down(192,384)
        self.down4 =Down(384,768)
        self.patch_merging1 = PatchMerging1(24, 96)
        self.patch_merging2 = PatchMerging1(48, 192)
        self.patch_merging3 = PatchMerging1(96, 384)
        self.patch_merging4 = PatchMerging1(192, 768)
        self.ch_down1 =Down_channel(192)
        self.ch_down2 =Down_channel(384)
        self.ch_down3 =Down_channel(768)
        self.ch_down4 =Down_channel(1536)
        self.channel_shuffle=channel_shuffle()
        self.simam1=simam_module(192)
        self.simam2=simam_module(384)
        self.simam3=simam_module(768)
        self.simam4=simam_module(1536)

        self.layer1 = BasicLayer(dim=96,  # 96, 192, 384, 768
                                 input_resolution=(56, 56),  # 56, 28, 14, 7
                                 depth=2,  # 2, 2, 6, 2
                                 num_heads=3,  # 3, 6, 12, 24
                                 window_size=self.window_size,  # 7
                                 mlp_ratio=self.mlp_ratio,  # 4.
                                 qkv_bias=self.qkv_bias,  # True
                                 qk_scale=self.qk_scale,  # None
                                 drop=self.drop_rate,  # 0.
                                 attn_drop=self.attn_drop_rate,  # 0.
                                 drop_path=dpr[0:2],  # [0:2], [2:4], [4:10], [10:12]
                                 norm_layer=self.norm_layer,
                                 downsample=PatchMerging,  # PatchMerging, PatchMerging, PatchMerging, None,
                                 use_checkpoint=False)

        self.layer2 = BasicLayer(dim=192,  # 96, 192, 384, 768
                                 input_resolution=(28, 28),  # 56, 28, 14, 7
                                 depth=2,  # 2, 2, 6, 2
                                 num_heads=6,  # 3, 6, 12, 24
                                 window_size=self.window_size,  # 7
                                 mlp_ratio=self.mlp_ratio,  # 4.
                                 qkv_bias=self.qkv_bias,  # True
                                 qk_scale=self.qk_scale,  # None
                                 drop=self.drop_rate,  # 0.
                                 attn_drop=self.attn_drop_rate,  # 0.
                                 drop_path=dpr[2:4],  # [0:2], [2:4], [4:10], [10:12]
                                 norm_layer=self.norm_layer,
                                 downsample=PatchMerging,  # PatchMerging, PatchMerging, PatchMerging, None,
                                 use_checkpoint=False)

        self.layer3 = BasicLayer(dim=384,  # 96, 192, 384, 768
                                 input_resolution=(14, 14),  # 56, 28, 14, 7
                                 depth=6,  # 2, 2, 6, 2
                                 num_heads=12,  # 3, 6, 12, 24
                                 window_size=self.window_size,  # 7
                                 mlp_ratio=self.mlp_ratio,  # 4.
                                 qkv_bias=self.qkv_bias,  # True
                                 qk_scale=self.qk_scale,  # None
                                 drop=self.drop_rate,  # 0.
                                 attn_drop=self.attn_drop_rate,  # 0.
                                 drop_path=dpr[4:10],  # [0:2], [2:4], [4:10], [10:12]
                                 norm_layer=self.norm_layer,
                                 downsample=PatchMerging,  # PatchMerging, PatchMerging, PatchMerging, None,
                                 use_checkpoint=False)

        self.layer4 = BasicLayer(dim=768,  # 96, 192, 384, 768
                                 input_resolution=(7, 7),  # 56, 28, 14, 7
                                 depth=2,  # 2, 2, 6, 2
                                 num_heads=24,  # 3, 6, 12, 24
                                 window_size=self.window_size,  # 7
                                 mlp_ratio=self.mlp_ratio,  # 4.
                                 qkv_bias=self.qkv_bias,  # True
                                 qk_scale=self.qk_scale,  # None
                                 drop=self.drop_rate,  # 0.
                                 attn_drop=self.attn_drop_rate,  # 0.
                                 drop_path=dpr[10:12],  # [0:2], [2:4], [4:6], [6:7]
                                 norm_layer=self.norm_layer,
                                 downsample=None,  # PatchMerging, PatchMerging, PatchMerging, None,
                                 use_checkpoint=False)
        self.norm = norm_layer(self.num_features)
        # self.decode = MOEHead(prescale_mlp_dims=[256, 256],  # new
        #                       # prescale_mlp_dims=[96, 96],  # new
        #                       # prescale_mlp_dims=[(256,256),(512,512),(1024,1024),(2048,2048)],  # new
        #                       prescale_mlp_final_act=True,
        #                       afterscale_mlp_dims=[256, 256],  # new
        #                       afterscale_mlp_final_act=True,
        #                       moe_mlp_dims=None,  # new
        #                       moe_conv_dims=[256],
        #                       in_channels=[192, 384, 768 , 1536],
        #                       in_index=[0, 1, 2, 3],
        #                       feature_strides=[4, 8, 16, 32],
        #                       channels=256,
        #                       dropout_ratio=0.1,
        #                       num_classes=4,
        #                       align_corners=False,
        #                       decoder_params=dict(embed_dim=256),
        #                       dim=256,
        #                       loss_decode=dict(type='StructurelLoss'))
        self.decode = MOEHead(prescale_mlp_dims=[96, 96],  # new
                              prescale_mlp_final_act=True,
                              afterscale_mlp_dims=[96, 96],  # new
                              afterscale_mlp_final_act=True,
                              moe_mlp_dims=None,  # new
                              moe_conv_dims=[96],
                              in_channels=[192, 384, 768, 1536],
                              in_index=[0, 1, 2, 3],
                              feature_strides=[4, 8, 16, 32],
                              channels=96,
                              dropout_ratio=0.1,
                              num_classes=5,
                              align_corners=False,
                              decoder_params=dict(embed_dim=96),
                              dim=96,
                              loss_decode=dict(type='StructurelLoss'))
        # self.apply(self._init_weights)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

      #Dencoder and Skip connection


    def transfer(self,x):
        B,L,C = x.shape
        H = int(math.sqrt(L))
        W = H
        x = x.view(B,H,W,C).permute(0,3,1,2)
        return x

    def transfer1(self,x):
        B,C,H,W = x.shape
        x = x.view(B,C,-1).permute(0,2,1)
        return x

    # def up_x4(self, x):
    #     H, W = self.patches_resolution
    #     B, L, C = x.shape
    #     assert L == H * W, "input features has wrong size"
    #
    #     if self.final_upsample == "expand_first":
    #         x = self.up(x)
    #         x = x.view(B, 4 * H, 4 * W, -1)
    #         x = x.permute(0, 3, 1, 2)  # B,C,H,W
    #         x = self.output(x)
    #
    #     return x

    def forward(self, x):
        x=self.inc(x)
        x_branchC = x[ :, 0::2, :, :]   #1,12,224,224
        x_branchT = x[ :, 1::2, :, :]   #1,12,224,224

        x_branchT = self.patch_embed(x_branchT)  # 1,3136,96
        x_branchC = self.down1(x_branchC)   #1,24,112,112
        conv_1 =self.patch_merging1(x_branchC)  #1,96,56,56
        trans_1 =self.transfer(x_branchT)   #1,96,56,56
        x = torch.cat([conv_1,trans_1],1)   #1,192,56,56
        x = self.channel_shuffle(x,2)   #1,192,56,56
        input_1 = x
        x1=self.simam1(input_1) #new
        t1=input_1  #visual
        # input_1 = self.ch_down1(x)   #1,96,56,56 input1

        if self.ape:
            x_branchT = x_branchT + self.absolute_pos_embed
        x_branchT = self.pos_drop(x_branchT)

        x_branchC = x[ :, 0::2, :, :]   #1,96,56,56
        x_branchT = x[ :, 1::2, :, :]   #1,96,56,56
        x_branchT = self.transfer1(x_branchT)
        x_branchT= self.layer1(x_branchT)  #1,784,192
        conv2 = self.down2(x_branchC)    #1,192,28,28
        # conv_2 = self.patch_merging1(conv2) #1,96,56,56
        trans2 = self.transfer(x_branchT)  #1,192,28,28
        x = torch.cat([conv2,trans2],1)
        x = self.channel_shuffle(x,2)   #1,384,28,28
        input_2 = x
        x2=self.simam2(input_2) #new
        t2=x    #visual

        x_branchC = x[:, 0::2, :, :]    #1,192,28,28
        x_branchT = x[:, 1::2, :, :]    #1,192,28,28
        x_branchT = self.transfer1(x_branchT)   #1,784,192
        x_branchT = self.layer2(x_branchT)  # 1,196,384
        conv3 = self.down3(x_branchC)   #1,384,14,14
        trans3 = self.transfer(x_branchT)  #1,384,14,14
        x = torch.cat([conv3,trans3],1)
        x = self.channel_shuffle(x,2)   #1,768,14,14
        input_3 = x
        x3=self.simam3(input_3) #new
        t3=x    #visual
        # input_3 = self.ch_down3(x)  #1,384,14,14

        x_branchC = x[:, 0::2, :, :]    #1,384,14,14
        x_branchT = x[:, 1::2, :, :]    #1,384,14,14
        x_branchT = self.transfer1(x_branchT)   #1,196,384
        x_branchT = self.layer3(x_branchT)  #1,49,768
        conv4 = self.down4(x_branchC)   #1,768,7,7
        # conv_4 = self.patch_merging3(conv3) #1,384,14,14
        trans4 = self.transfer(x_branchT)  #1,768,7,7
        x = torch.cat([conv4,trans4],1) #1,1536,7,7
        input_4 = x
        x4=self.simam4(input_4)
        # t4=x    #visual
        # input_4 = self.ch_down4(x)        #5.327G 23.816M
        x = self.decode([x1,x2,x3,x4])
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

logger = logging.getLogger(__name__)

class Shufflemoe_config():
    def __init__(self):
        self.patch_size = 4
        self.in_chans = 3
        self.num_classes = 4
        self.embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.window_size = 7
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.
        self.drop_path_rate = 0.1
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False

class Shufflemoe(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(Shufflemoe, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.shuffle_moe = SwinTransformerSys(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=4,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads = [3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.1,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.shuffle_moe(x)
        return logits

    def load_from(self, pretrained_path):
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.shuffle_moe.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            model_dict = self.shuffle_moe.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
            msg = self.shuffle_moe.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

if __name__ == "__main__":
    with torch.no_grad():
        input = torch.rand(1, 3, 224, 224).to('cuda:0')
        model = SwinTransformerSys().to('cuda:0')
        out_result = model(input)
        print(out_result.shape)
        flops1, params1 = profile(model, (input,))
        flops1, params1 = clever_format([flops1, params1], "%.3f")
        print(flops1, params1)