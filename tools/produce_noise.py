import glob
import os

import numpy as np
import cv2
import math
import random
from PIL import ImageEnhance


#运动模糊噪声
def motion_blur(image, degree=25, angle=45):
    image = np.array(image)
    # image = cv2.imread(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    # cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def gaosi(image):
    # 设置高斯分布的均值和方差
    mean = 0
    # 设置高斯分布的标准差
    sigma = 25
    # 根据均值和标准差生成符合高斯分布的噪声
    # gauss = np.random.normal(mean, sigma, (1024, 1280, 3))
    gauss = np.random.normal(mean, sigma, (1024, 2048, 3))
    # 给图片添加高斯噪声
    noisy_img = image + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img

def jiaoyan(image):
    # 设置添加椒盐噪声的数目比例
    s_vs_p = 0.5
    # 设置添加噪声图像像素的数目
    amount = 0.04
    noisy_img = np.copy(image)
    # 添加salt噪声
    num_salt = np.ceil(amount * image.size * s_vs_p)
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = [255, 255, 255]
    # 添加pepper噪声
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    # 设置添加噪声的坐标位置
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_img[coords[0], coords[1], :] = [0, 0, 0]
    return noisy_img

def bosong(image):
    # 计算图像像素的分布范围
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    # 给图片添加泊松噪声
    noisy_img = np.random.poisson(image * vals) / float(vals)
    return noisy_img

def apeckle(image):
    # 随机生成一个服从分布的噪声
    # gauss = np.random.randn(1024, 1280, 3)
    gauss = np.random.randn(1024, 2048, 3)
    # 给图片添加speckle噪声
    noisy_img = image + image * gauss
    # 归一化图像的像素值
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img

def jiawu(img):
    img_f = img
    (row, col, chs) = img.shape
    A = 250  # 亮度
    beta = 0.03  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

def liangdu(img):
    rows, cols = img.shape[:2]

    # 设置中心点
    centerX = rows / 1.5
    centerY = cols / 1.5
    print(centerX, centerY)
    radius = min(centerX, centerY)
    print(radius)

    # 设置光照强度
    strength = 150

    # 图像光照特效
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            # 获取原始图像
            B = img[i, j][0]
            G = img[i, j][1]
            R = img[i, j][2]
            if (distance < radius * radius):
                # 按照距离大小计算增强的光照值
                result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                B = img[i, j][0] + result
                G = img[i, j][1] + result
                R = img[i, j][2] + result
                # 判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                img[i, j] = np.uint8((B, G, R))
            else:
                img[i, j] = np.uint8((B, G, R))
    return img
#-----------------------------------------------------------------
    # lightness = 80   #亮度-100~+100
    # saturation =50   #饱和度-100~+100
    # MAX_VALUE = 100
    # image = img.astype(np.float32) / 255.0
    #
    # # 颜色空间转换 BGR转为HLS
    # hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #
    # # 1.调整亮度（线性变换)
    # hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    # hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # # 饱和度
    # hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    # hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # # HLS2BGR
    # lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    # lsImg = lsImg.astype(np.uint8)
    # return  lsImg


if __name__ == '__main__':

    """edv2018"""
    # image_path = "/data2/hyb/DataSurgery/endovis2018_train2235_val997_test997/leftImg8bit/test/"
    # # new_path = "/data2/hyb/DataSurgery/endovis2018_train2235_val997_test997/leftImg8bit/testmotionblur/"
    # # new_path = "/data2/hyb/DataSurgery/endovis2018_train2235_val997_test997/leftImg8bit/testjiawu/"
    # new_path = "/data2/hyb/DataSurgery/endovis2018_train2235_val997_test997/leftImg8bit/test_motionblur_jiawu/"
    # if not os.path.isdir(new_path):
    #     os.mkdir(new_path)
    # test_dirs = [os.path.join(image_path, "seq{}test".format(i)) for i in range(1, 5)]
    # for test_dir in test_dirs:
    #     new_test_dir = test_dir.replace(image_path, new_path)
    #     if not os.path.isdir(new_test_dir):
    #         os.mkdir(new_test_dir)
    #     for pic in os.listdir(test_dir):
    #         img = cv2.imread(os.path.join(test_dir,pic), 1)
    #         img_ = motion_blur(img)
    #         img_ = jiawu(img_) #motion_blur  #jiaoyan   #gaosi   #bosong   #apeckle #liangdu
    #         cv2.imwrite(os.path.join(new_test_dir, pic),img_)
    # print('Finish')

    """Cadis"""
    # image_path = "/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test"
    # # new_path = "/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test_motionblur/"
    # new_path = "/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test_jiawu/"
    # # new_path = "/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test_motionblur_jiawu/"
    # if not os.path.isdir(new_path):
    #     os.mkdir(new_path)
    # test_dirs = [os.path.join(image_path, "Video{}".format(i)) for i in [2, 12, 22]]
    # for test_dir in test_dirs:
    #     new_test_dir = test_dir.replace(image_path, new_path)
    #     if not os.path.isdir(new_test_dir):
    #         os.mkdir(new_test_dir)
    #     for pic in os.listdir(test_dir):
    #         img = cv2.imread(os.path.join(test_dir, pic), 1)
    #         # img_ = motion_blur(img)
    #         img_ = jiawu(img)  # motion_blur  #jiaoyan   #gaosi   #bosong   #apeckle #liangdu
    #         cv2.imwrite(os.path.join(new_test_dir, pic), img_)
    # print('Finish')

    """MILS"""
    image_path = "/data2/hyb/DataSurgery/Lapavis_train2250_test750/leftImg8bit/test"
    # new_path = "/data2/hyb/DataSurgery/Lapavis_train2250_test750/leftImg8bit/test_motionblur/"
    new_path = "/data2/hyb/DataSurgery/Lapavis_train2250_test750/leftImg8bit/test_jiawu/"
    # new_path = "/data2/hyb/DataSurgery/CaDIS_train3550_val534_test586_task3/leftImg8bit/test_motionblur_jiawu/"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    for pic in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, pic), 1)
        # img_ = motion_blur(img)
        img_ = jiawu(img)  # motion_blur  #jiaoyan   #gaosi   #bosong   #apeckle #liangdu
        cv2.imwrite(os.path.join(new_path, pic), img_)
    print('Finish')



