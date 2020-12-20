checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/user1/文档/mmdetection-master/work_dirs/cascade_mask_rcnn_r101_fpn_1x_coco/latest.pth'
workflow = [('train', 1)]
