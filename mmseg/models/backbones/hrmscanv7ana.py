# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import math
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.ops import Upsample, resize
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck
import warnings
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)


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


class Dev_AttentionModule(BaseModule):
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

        # attn = attn + attn_0 + attn_1 + attn_2
        attn = attn + attn_1 + attn_2 + attn_3

        attn = self.conv3(attn)

        return attn * u


class Ana_AttentionModule(BaseModule):
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

        # attn = attn + attn_0 + attn_1 + attn_2
        attn = attn + attn_1 + attn_2 + attn_3
        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # self.norm_1 = build_norm_layer(dict(type='BN', requires_grad=True), dim)[1]
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        # self.activation = nn.GELU()
        self.activation = nn.ReLU()
        self.dev_spatial_gating_unit = Dev_AttentionModule(dim)
        self.ana_spatial_gating_unit = Ana_AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        # self.norm_2 = build_norm_layer(dict(type='BN', requires_grad=True), dim)[1]

    def forward(self, x):
        shorcut = x.clone()
        # x = self.norm_1(x)
        x = self.proj_1(x)
        x = self.activation(x)

        x_dev = self.dev_spatial_gating_unit(x)
        x_dev = self.proj_2(x_dev)

        x_ana = self.ana_spatial_gating_unit(x)
        x_ana = self.proj_2(x_ana)

        # x = shorcut + x_dev + x_ana
        x = shorcut + x_ana
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
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    # def forward(self, x, H, W):
    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class HRModule(BaseModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 block_init_cfg=None,
                 init_cfg=None):
        super(HRModule, self).__init__(init_cfg)
        self.block_init_cfg = block_init_cfg
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        """Check branches configuration."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS(' \
                        f'{len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS(' \
                        f'{len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS(' \
                        f'{len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Build one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=self.block_init_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=self.block_init_cfg))

        return Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Build multiple branch."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            # we set align_corners=False for HRNet
                            Upsample(
                                scale_factor=2**(j - i),
                                mode='bilinear',
                                align_corners=False)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + resize(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=False)
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class HRMscanv7ana(BaseModule):
    """hrmscanv7，只做ana_attn"""

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 depths=[1, 3, 3, 3],
                 num_stages=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 frozen_stages=-1,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super(HRMscanv7ana, self).__init__(init_cfg)

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
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

        # Assert configurations of 4 stages are in extra
        assert 'stage1' in extra and 'stage2' in extra \
               and 'stage3' in extra and 'stage4' in extra
        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages
        self.depths = depths
        self.num_stages = num_stages

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=multiscale_output)

        self._freeze_stages()

        # Fusion mscan stage
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            fusion_mscan_block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                                                      drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                                                for j in range(depths[i])])
            batch_norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += depths[i]
            setattr(self, f"fusion_mscan_block{i + 1}", fusion_mscan_block)
            setattr(self, f"batch_norm{i + 1}", batch_norm)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))

        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=block_init_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=block_init_cfg))

        return Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make each stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))

        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg))

        return Sequential(*hr_modules), in_channels

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:

            self.norm1.eval()
            self.norm2.eval()
            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = getattr(self, f'layer{i}')
                t = getattr(self, f'transition{i}')
            elif i == 4:
                m = getattr(self, f'stage{i}')
            else:
                m = getattr(self, f'stage{i}')
                t = getattr(self, f'transition{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            t.eval()
            for param in t.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        """stage2"""
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        """stage3"""
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        """stage4"""
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        """Fusion mscan stage"""
        outs = []
        for i in range(len(y_list)):
            x = y_list[i]
            fusion_mscan_block = getattr(self, f"fusion_mscan_block{i + 1}")
            batch_norm = getattr(self, f"batch_norm{i + 1}")
            for blk in fusion_mscan_block:
                x = blk(x)
            x = batch_norm(x)
            outs.append(x)

        # return y_list
        return outs


    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(HRMscanv7ana, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
