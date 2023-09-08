_base_ = './upernet_r50_edv2018.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
