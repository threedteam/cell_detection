# _base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
# model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(pretrained='https://download.pytorch.org/models/densenet121-a639ec97.pth',
             backbone=dict(depth=121,type='DenseNet',num_stages=4))