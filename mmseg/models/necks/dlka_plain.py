import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from ..builder import NECKS


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
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=int(dim/3))

        # self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))

        # self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))

        # self.conv3_1 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))
        # self.conv4 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/2))

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        # attn_1_1 = self.conv1_1(attn)
        # attn_1_2 = self.conv1_2(attn)
        # attn_1 = attn_1_1 + attn_1_2
        attn_1 = self.conv1(attn)

        # attn_2_1 = self.conv2_1(attn)
        # attn_2_2 = self.conv2_2(attn)
        # attn_2 = attn_2_1 + attn_2_2
        attn_2 = self.conv2(attn)

        # attn_3_1 = self.conv3_1(attn)
        # attn_3_2 = self.conv3_2(attn)
        # attn_3 = attn_3_1 + attn_3_2
        attn_3 = self.conv3(attn)
        # attn_4 = self.conv4(attn)
        attn = attn + attn_1 + attn_2 + attn_3

        attn = self.conv4(attn)

        return attn * u


class ana_block(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=int(dim/3))

        # self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        # self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))

        # self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        # self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))

        # self.conv3_1 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/3))
        # self.conv4 = nn.Conv2d(dim, dim, 3, 1, 1, groups=int(dim/2))

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        # attn_1 = self.conv1_1(attn)
        # attn_1 = self.conv1_2(attn_1)
        #
        # attn_2 = self.conv2_1(attn)
        # attn_2 = self.conv2_2(attn_2)
        #
        # attn_3 = self.conv3_1(attn)
        # attn_3 = self.conv3_2(attn_3)
        #
        # attn = attn + attn_1 + attn_2 + attn_3
        # attn = self.conv3(attn)

        attn_1 = self.conv1(attn)
        attn_2 = self.conv2(attn)
        attn_3 = self.conv3(attn)
        # attn_4 = self.conv4(attn)
        attn = attn + attn_1 + attn_2 + attn_3

        attn = self.conv4(attn)

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


@NECKS.register_module()
class DLKA_Plain(BaseModule):

    def __init__(self,
                 embed_dims,
                 mlp_ratios,
                 drop_rate=0.0,
                 drop_path_rate=0.1,
                 depths=[1, 2, 4, 2],
                 num_stages=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        super(DLKA_Plain, self).__init__(init_cfg=init_cfg)

        self.conv_cfg = conv_cfg
        self.num_stages = num_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(self.num_stages):
            dlka_block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate,
                                              drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                                        for j in range(depths[i])])
            batch_norm = build_norm_layer(norm_cfg, embed_dims[i])[1]
            cur += depths[i]
            setattr(self, f"dlka_block{i + 1}", dlka_block)
            setattr(self, f"batch_norm{i + 1}", batch_norm)

    def forward(self, inputs):

        outs = []
        if range(len(inputs)) == 4:
            inputs = inputs
        else:
            inputs = inputs[-4:]

        for i in range(len(inputs)):
            x = inputs[i]
            dlka_block = getattr(self, f"dlka_block{i + 1}")
            batch_norm = getattr(self, f"batch_norm{i + 1}")
            for blk in dlka_block:
                x = blk(x)
            x = batch_norm(x)
            outs.append(x)

        return outs