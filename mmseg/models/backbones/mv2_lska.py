# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from ..utils import InvertedResidual, make_divisible
import torch.nn as nn
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..builder import BACKBONES
import warnings
from mmcv.cnn.bricks import DropPath


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ins_block(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1_1 = self.conv1_1(attn)
        attn_1_2 = self.conv1_2(attn)
        attn_1 = attn_1_1 + attn_1_2

        attn_2_1 = self.conv2_1(attn)
        attn_2_2 = self.conv2_2(attn)
        attn_2 = attn_2_1 + attn_2_2

        attn_3_1 = self.conv3_1(attn)
        attn_3_2 = self.conv3_2(attn)
        attn_3 = attn_3_1 + attn_3_2

        attn = attn + attn_1 + attn_2 + attn_3

        attn = self.conv3(attn)

        return attn * u


class ana_block(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)

        attn = attn + attn_1 + attn_2 + attn_3
        attn = self.conv3(attn)

        return attn * u


class LSKAttention(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.ReLU()
        self.ins_block = ins_block(dim)
        self.ana_block = ana_block(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)

        x_ins = self.ins_block(x)
        x_ins = self.proj_2(x_ins)

        x_ana = self.ana_block(x)
        x_ana = self.proj_2(x_ana)

        x = shorcut + x_ins + x_ana
        return x


class Block(BaseModule):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 # act_layer=nn.GELU,
                 act_layer=nn.ReLU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = LSKAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


@BACKBONES.register_module()
class MobileNetV2LSKA(BaseModule):

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4],
                     [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self,
                 widen_factor=1.,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 dilations=(1, 1, 1, 1, 1, 1, 1),
                 out_indices=(1, 2, 4, 6),
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 num_stages=4,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 depths=[1, 3, 3, 3],
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(MobileNetV2LSKA, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 7). But received {index}')

        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.num_stages = num_stages

        self.in_channels = make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        # lska module
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            lska_block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                                              drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                                        for j in range(depths[i])])
            batch_norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += depths[i]
            setattr(self, f"lska_block{i + 1}", lska_block)
            setattr(self, f"batch_norm{i + 1}", batch_norm)

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        for i in range(len(outs)):
            x = outs[i]
            lska_block = getattr(self, f"lska_block{i + 1}")
            batch_norm = getattr(self, f"batch_norm{i + 1}")
            for blk in lska_block:
                x = blk(x)
            x = batch_norm(x)
            outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV2LSKA, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
