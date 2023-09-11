# LSKANet: Long Strip Kernel Attention Network for Robotic Surgical Scene Segmentation
by Min Liu, Yubin Han, Jiazheng Wang, Can Wang*, Yaonan Wang, and Erik Meijering.
## Introduction
* We proposed a surgical scene segmentation network named **Long Strip Kernel Attention network (LSKANet)**, which includes two newly designed modules, **Dual-block Large Kernel Attention module (DLKA)** and **Multiscale Affinity Feature Fusion module (MAFF)**. Besides, the hybrid loss with **Boundary Guided Head (BGH)** is proposed to help the network segment indistinguishable boundaries effectively. Our LSKANet achieves new state-of-the-art results on three datasets with different surgical scenes (Endovis2018, CaDIS, and MILS) with relative improvements of 2.6%, 1.5%, and 3.4% mIoU, respectively.

![farmework](https://github.com/YubinHan73/LSKANet/assets/71008581/de64f69d-df2e-457b-8aff-de4e1a501fa0)

## Dataset
* We evaluate the proposed LSKANet on three datasets：[Endovis2018](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Downloads/), [CaDIS](https://ieee-dataport.org/open-access/cataracts), and a self-built dataset called **Minimally Invasive Laparoscopic Surgery dataset (MILS)**, which will be released in the future.

## Results
We provide some visualization resluts here, more resluts can be found in paper.
* **Visualization results on Endovis2018 (12 classes)**

![image](https://github.com/YubinHan73/LSKANet/assets/71008581/8a534745-38b9-4dd6-a51b-3c52c03d9829)

* **Visualization results on CaDIS (Task Ⅲ, 25 classes)**

![image](https://github.com/YubinHan73/LSKANet/assets/71008581/a64e29fb-455d-481b-a3f7-20295c159cc4)

* **Visualization results on MILS (8 classes)**

![image](https://github.com/YubinHan73/LSKANet/assets/71008581/7cb49452-d8f6-4a2f-a462-06b00080252a)

## Usage
### Requirements
We used these packages/versions in the development of this project.
```
* PyTorch 1.10.0
* torchvision 0.12.0
* mmcv 1.6.1
* mmsegmentation 0.24.1
* opencv-python 4.5.3
```
### Training process
Before training, please download the dataset you need and rename them following `mmseg/datasets/endovis2018.py` and `mmseg/datasets/cadis.py`.
1. Switch folder `cd ./tools/`
2. Use `python train.py` to start the training
3. Parameter setting and training script refer to `/work_dirs/0_LSKANet/LSKANet_XXXX.py`

### Test & Visualization
1. Use `python test.py` to start the inferencing
2. Visualization results can be found in `/tools/test_out/`

## Acknowledgements
We build our code on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). Thanks original authors for their impressive work!

## Questions
For further question about the code or paper, please contact Yubin Han:15073176834@163.com.
