
import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def heatmap(feats, labels=None, type='PCA', out_dim=3, out_shape=None, norm255=True):
    '''
    feats: torch.tensor(B, C, H1, W1)
    labels: torch.tensor(B, H2, W2)
    type: 'PCA', 'LDA', 计算方法
    out_dim: 输出维度，3方便可视化
    out_shape: [new_H, new_W]
    norm255: 转成0-255还是0-1

    output: torch.tensor(B, out_dim, new_H, new_W)
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

    image = (image - mins) / maxs
    image = image.reshape(B, H, W, out_dim).permute(0, 3, 1, 2)
    image = F.interpolate(image, (new_H, new_W), mode='bilinear', align_corners=True)
    if norm255:
        image = 256 * image
        image = image.to(torch.uint8)

    return image


def class_highlight_heatmap(feats, labels, highlight_class=[1,], out_shape=None, cmap='bwr', rgb=True):
    '''
    feats: torch.tensor(B, C, H1, W1)
    labels: torch.tensor(B, H2, W2)
    highlight_class: 需要展示的类别，会被作为一类整体对待，剩余的作为另一类
    out_shape: [new_H, new_W]

    output: torch.tensor(B, 3, new_H, new_W), 0-1 or 0-255
    '''
    B, C, H, W = feats.shape
    if out_shape is not None:
        assert len(out_shape) == 2
        new_H, new_W = out_shape
    else:
        new_H, new_W = H, W

    # 制作新的label
    new_labels = torch.zeros_like(labels)
    for _label in highlight_class:
        new_labels[labels==_label] = 1

    if new_labels.sum() == 0:
        if rgb and cmap is not None:
            return torch.zeros(B, 3, new_H, new_W, dtype=torch.uint8)
        else:
            return torch.zeros(B, 1, new_H, new_W, dtype=torch.double)

    new_labels = F.interpolate(new_labels.to(torch.double).unsqueeze(1), (H, W), mode='nearest')
    new_labels = new_labels.to(torch.long).squeeze(1)

    # LDA
    image = heatmap(feats, new_labels, type='LDA', out_dim=1, norm255=False)

    mean0 = image[new_labels.unsqueeze(1) == 0].mean()
    mean1 = image[new_labels.unsqueeze(1) == 1].mean()
    # if mean0 > mean1:
    #     image = 1 - image

    # 分段归一化
    eps = 1e-8
    min1 = (image[new_labels.unsqueeze(1) == 1]).min()
    image[new_labels.unsqueeze(1).to(bool)] = (image[new_labels.unsqueeze(1).to(bool)] - min1) / (1 - min1 + eps) * 0.5 + 0.5
    image[~new_labels.unsqueeze(1).to(bool)] = image[~new_labels.unsqueeze(1).to(bool)] / (min1 + eps) * 0.5

    image = F.interpolate(image, (new_H, new_W), mode='bilinear', align_corners=True)

    # gray to rgb
    if rgb and cmap is not None:
        # 选择colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html#classes-of-colormaps
        colormap = plt.get_cmap(cmap)
        image = image.numpy()
        image = colormap(image, bytes=True).astype(np.uint8)[..., :3]
        image = torch.from_numpy(image).squeeze(1).permute(0, 3, 1, 2)

    return image


if __name__ == '__main__':
    feats = torch.rand(2, 12, 128, 128)
    labels = torch.randint(0, 12, (2, 256, 256))

    image = heatmap(feats, labels, 'PCA', out_shape=[256, 256])
    print(f'The shape of image after PCA is: {image.shape}')

    image = heatmap(feats, labels, 'LDA', out_shape=[256, 256])
    print(f'The shape of image after LDA is: {image.shape}')

    image = class_highlight_heatmap(feats, labels, highlight_class=[1, 2, 3], out_shape=[256, 256])
    print(f'The shape of image after class highlight is: {image.shape}')
