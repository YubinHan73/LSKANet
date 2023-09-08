import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmengine.model import revert_sync_batchnorm


def main():
    config = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/work_dirs/0_LSKANet/LSKANet_hr48_cadist3_new.py'
    checkpoint = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/work_dirs/0_LSKANet/LSKANet_hr48_cadist3_new/best_mIoU_epoch_108.pth'
    # config = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/work_dirs/0_LSKANet/LSKANet_hr48_LSKA_cadist3.py'
    # checkpoint = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/work_dirs/0_LSKANet/LSKANet_hr48_LSKA_cadist3/best_mIoU_epoch_100.pth'
    img_path = '/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test/Video22/Video22_frame019920_leftImg8bit.png'
    save_path = '/data2/hyb/SegNetwork_other/SegNeXt-main/tools/cam_out_cadis/{}'.format(
        img_path.split('/')[-1].split('_')[1])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # model = models.mobilenet_v3_large(pretrained=True)
    register_all_modules()

    model = init_model(config, checkpoint, device='cpu')
    model = revert_sync_batchnorm(model)
    target_layers = [model.features[-1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    # img_path = "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
