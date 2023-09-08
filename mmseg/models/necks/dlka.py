import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from ..builder import NECKS
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tools.show_feature_map import class_highlight_heatmap, label


def heatmap(feats, labels=None, type='PCA', out_dim=3, out_shape=None):
    '''
    feats: torch.tensor(B, C, H, W)
    labels: torch.tensor(B, H, W)
    type: 'PCA', 'LDA', 计算方法
    out_dim: 输出维度，3方便可视化
    out_shape: [new_H, new_W]

    output: torch.tensor(B, out_dim, H1, W1), 0-255
    '''

    B, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
    if labels is not None:
        labels = F.interpolate(labels.to(torch.double).unsqueeze(1), (H, W), mode='nearest')
        labels = labels.reshape(-1)

    if type == 'PCA':
        image = PCA(n_components=out_dim).fit_transform(feats)
    elif type == 'LDA':
        assert labels is not None
        image = LDA(n_components=out_dim).fit_transform(feats, labels)
    else:
        raise NotImplementedError
    image = torch.from_numpy(image)

    if out_shape is not None:
        assert len(out_shape) == 2
        new_H, new_W = out_shape
    else:
        new_H, new_W = H, W

    mins = image.min(dim=0, keepdim=True)[0]
    maxs = image.max(dim=0, keepdim=True)[0]

    image = 256 * (image - mins) / maxs
    image = image.reshape(B, H, W, out_dim).permute(0, 3, 1, 2)
    image = F.interpolate(image, (new_H, new_W), mode='bilinear', align_corners=True)
    image = image.to(torch.uint8)

    return image


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

        # x_fea_in = torch.cat((shorcut, x_ins, x_ana), dim=0)
        # x_fea_out = class_highlight_heatmap(x_fea_in.cpu(), labels=label, highlight_class=[0, 4, 5, 6], out_shape=None, cmap='bwr', rgb=True)
        # x_fea_outs = torch.split(x_fea_out, 1, dim=0)
        # array_x = np.transpose(np.squeeze(x_fea_outs[0].numpy()), (1, 2, 0))
        # array_x_ins = np.transpose(np.squeeze(x_fea_outs[1].numpy()), (1, 2, 0))
        # array_x_ana = np.transpose(np.squeeze(x_fea_outs[0].numpy()), (1, 2, 0))

        # cam_x = class_highlight_heatmap(shorcut.cpu(), labels=label, highlight_class=[0, 4, 5, 6], out_shape=None, cmap='bwr', rgb=True)
        # array_x = cam_x.numpy()
        # array_x = np.transpose(np.squeeze(array_x), (1, 2, 0))
        # # image_cam_x =  Image.fromarray(array_x)
        # # image_cam_x.save('/data2/hyb/SegNetwork_other/SegNeXt-main/tools/test_out/cam_x.png')
        # #
        # cam_x_ins = class_highlight_heatmap(x_ins.cpu(), labels=label, highlight_class=[0, 4, 5, 6], out_shape=None, cmap='bwr', rgb=True)
        # array_x_ins = cam_x_ins.numpy()
        # array_x_ins= np.transpose(np.squeeze(array_x_ins), (1, 2, 0))
        # # image_cam_x_ins = Image.fromarray(array_x_ins)
        # # image_cam_x_ins.save('/data2/hyb/SegNetwork_other/SegNeXt-main/tools/test_out/cam_x_ins.png')
        # #
        # cam_x_ana = class_highlight_heatmap(x_ana.cpu(), labels=label, highlight_class=[0, 4, 5, 6], out_shape=None, cmap='bwr', rgb=True)
        # array_x_ana = cam_x_ana.numpy()
        # array_x_ana = np.transpose(np.squeeze(array_x_ana), (1, 2, 0))
        # # image_cam_x_ana = Image.fromarray(array_x_ana)
        # # image_cam_x_ana.save('/data2/hyb/SegNetwork_other/SegNeXt-main/tools/test_out/cam_x_ana.png')
        #
        # fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(array_x)
        # axes[0].axis('off')
        # axes[0].set_title('Image 1')
        #
        # axes[1].imshow(array_x_ins)
        # axes[1].axis('off')
        # axes[1].set_title('Image 2')
        #
        # axes[2].imshow(array_x_ana)
        # axes[2].axis('off')
        # axes[2].set_title('Image 3')
        #
        # # 调整子图之间的间距
        # plt.tight_layout()
        # plt.show()
        #
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
class DLKA(BaseModule):

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
        super(DLKA, self).__init__(init_cfg=init_cfg)

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