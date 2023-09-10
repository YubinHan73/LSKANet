import argparse
import cv2
import numpy as np
import torch
from torchvision import models
# from mmseg.apis.inference import inference_segmentor, init_segmentor
# import mmcv
# from mmseg.apis.inference import inference_segmentor, init_segmentor
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS

config = '/data2/hyb/SegNetwork_other/LSKANet_main/tools/work_dirs/0_LSKANet/LSKANet_cadist3.py'
checkpoint = '/data2/hyb/SegNetwork_other/LSKANet_main/tools/work_dirs/0_LSKANet/LSKANet_hr48_cadist3_new/best_mIoU_epoch_108.pth'
img_path = '/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test/Video22/Video22_frame019920_leftImg8bit.png'
label_path = img_path.replace('leftImg8bit', 'gtFine').split('.')[0] + '_labelIds.png'
transform = transforms.Compose([transforms.ToTensor()])
label = Image.open(label_path).convert('L')
label_np = np.array(label)
label = torch.from_numpy(label_np).unsqueeze(0)
# label = torch.cat((label, label, label), dim=0)
img = transform(Image.open(img_path))

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

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

    image = (image - mins) / (maxs - mins)
    image = image.reshape(B, H, W, out_dim).permute(0, 3, 1, 2)
    image = F.interpolate(image, (new_H, new_W), mode='bilinear', align_corners=True)
    if norm255:
        image = 256 * image
        image = image.to(torch.uint8)

    return image

def class_highlight_heatmap(feats, labels=label, highlight_class=[0,4,5,6], out_shape=None, cmap='bwr', rgb=True):
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
    if mean0 > mean1:
        image = 1 - image

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

if __name__ == "__main__":
    model = init_segmentor(config, checkpoint, device='cpu')
    result = inference_segmentor(model, img_path)
