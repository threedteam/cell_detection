from mmdet.apis import init_detector, inference_detector
import mmcv

config_file =  '/home/user1/文档/mmdetection-master/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'
checkpoint_file = '/media/user1/95ef234c-de2c-4164-9379-afd5a9fcfead/work_dirs/cascade_mask_dense/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/user1/文档/mmdetection-master/car.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result_8.bmp')



# from mmdet.apis import init_detector, inference_detector
# from mmdet.apis import inference_detector, init_detector, show_result_pyplot
# from mmdet.models.detectors import base
# import pycocotools.mask as mask_util
# import os.path as osp
# import pickle
# import shutil
# import tempfile
#
# import mmcv
# import numpy as np
# import pycocotools.mask as mask_util
# import torch
# import torch.distributed as dist
# from mmcv.runner import get_dist_info
#
# from mmdet.core import tensor2imgs
# from mmdet.models.detectors.base import show_result
# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# import os
# import mmcv
# import sys
#
# def saving():
#
#     config_file = '/home/user1/文档/mmdetection-master/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'
#     checkpoint_file = '/home/user1/文档/mmdetection-master/work_dirs/cascade_mask_dense/epoch_200.pth'
#
#     imge = '/home/user1/文档/mmdetection-master/svar87.BMP'
#     #img = sys.argv[1]
#     model1 = init_detector(config_file, checkpoint_file, device='cuda:0')
#     result1 = inference_detector(model1, imge)
#     if isinstance(result1, tuple):
#         bbox_results, mask_results = result1
#         encoded_mask_results = encode_mask_results(mask_results)
#         result1 = bbox_results, encoded_mask_results
#     print(type(imge))
#     print(type(result1))
#     img = '/home/user1/文档/mmdetection-master/svar87.BMP'
#     model1.show_result(img, result1, out_file='outpic1.BMP')#
#     print('over')
#
#
# def encode_mask_results(mask_results):
#     if isinstance(mask_results, tuple):  # mask scoring
#         cls_segms, cls_mask_scores = mask_results
#     else:
#         cls_segms = mask_results
#     num_classes = len(cls_segms)
#     encoded_mask_results = [[] for _ in range(num_classes)]
#     for i in range(len(cls_segms)):
#         for cls_segm in cls_segms[i]:
#             encoded_mask_results[i].append(
#                 mask_util.encode(
#                     np.array(
#                         cls_segm[:, :, np.newaxis], order='F',
#                         dtype='uint8'))[0])  # encoded with RLE
#     if isinstance(mask_results, tuple):
#         return encoded_mask_results, cls_mask_scores
#     else:
#         return encoded_mask_results
# #
# # config_file = '/home/usr1/文档/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'
# # checkpoint_file = '/media/usr1/95ef234c-de2c-4164-9379-afd5a9fcfead/usr1/mmd_result/newmmd6300/epoch_40.pth'
# #
# # # build the model from a config file and a checkpoint file
# # model = init_detector(config_file, checkpoint_file, device='cuda:0')
# #
# # # test a single image and show the results
# # #img = 'light_dysplastic_2.BMP'  # or
# # img = mmcv.imread('light_dysplastic_2.BMP') #which will only load it once
# # result = inference_detector(model, img)
# # # # visualize the results in a new window
# # # print(result)
# # # mask_result = encode_mask_results(result)
# # #model.show_result(img, mask_result)
# # # or save the visualization results to image files
# # #show_result(img, result, out_file='result.jpg')
# # show_result_pyplot(model, img, result, score_thr=0.5)
#
#
#
# t=saving()