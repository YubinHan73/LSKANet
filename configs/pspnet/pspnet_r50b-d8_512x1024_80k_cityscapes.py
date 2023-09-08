_base_ = './pspnet_r50_edv2018.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
