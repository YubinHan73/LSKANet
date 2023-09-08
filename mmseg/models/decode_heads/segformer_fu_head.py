import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import Upsample, resize
from mmcv.cnn import build_conv_layer, build_norm_layer


@HEADS.register_module()
class SegformerFUHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.conv_cfg = None
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        sum_channels = 0
        for channel in self.in_channels:
            sum_channels += channel

        self.fusion_conv = ConvModule(
            in_channels=sum_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        """定义特征图融合的gateblock"""
        self.gate_in_blocks = nn.ModuleList()
        for i in range(num_inputs):
            self.gate_in_blocks.append(
                ConvModule(self.in_channels[i], self.in_channels[0], kernel_size=1, stride=1, padding=0,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None))

        self.gate_aff_blocks = nn.ModuleList()
        for i in range(num_inputs):
            self.gate_aff_blocks.append(
                ConvModule(self.in_channels[0], 1, kernel_size=1, stride=1, padding=0,
                           conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None))

        self.relu = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()
        self.bottleneck = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        x = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        """进行特征图的融合"""
        for i in range(len(x)):
            if i == 0:
                continue
            else:
                in_feat = self.gate_in_blocks[i](x[i])
                aff_ = self.relu(x[0] + in_feat)
                aff = self.gate_aff_blocks[i](aff_)
                aff = self.Sigmoid(aff)
                x[i] = x[i] * aff
        # out = self.bottleneck(torch.cat(x, dim=1))
        out = self.fusion_conv(torch.cat(x, dim=1))
        # outs = []
        # for idx in range(len(inputs)):
        #     x = inputs[idx]
        #
        #     conv = self.convs[idx]
        #     outs.append(
        #         resize(
        #             input=conv(x),
        #             size=inputs[0].shape[2:],
        #             mode=self.interpolate_mode,
        #             align_corners=self.align_corners))

        # out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
