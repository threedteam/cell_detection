import os
import json
import numpy as np
import glob
import shutil
# from sklearn.model_selection import train_test_split
from collections import OrderedDict
import random
import math
# np.random.seed(41)

# 0为背景
classname_to_id = {"carcinoma_in_situ": 7, 'light_dysplastic': 4,
                   'moderate_dysplastic': 5, 'normal_columnar': 3,
                   'normal_intermediate': 2, 'normal_superficiel': 1,
                   'severe_dysplastic': 6}
# 每类新增图片数
class_to_num = {"carcinoma_in_situ": 866, 'light_dysplastic': 834,
                   'moderate_dysplastic': 870, 'normal_columnar': 918,
                   'normal_intermediate': 946, 'normal_superficiel': 942,
                   'severe_dysplastic': 819}
# 数据增强类型
data_augment = ['translation', 'rotation', 'scaling', 'feature_standardization',
                'zca_whitening', 'vertical_flip', 'horizontal_flip']


class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape, json_path)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = OrderedDict()
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = OrderedDict()
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = OrderedDict()
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['file_name'] = os.path.basename(path).replace(".json", ".BMP")
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, json_path):
        # 图像面积的路径
        area_path = json_path.replace('labelme', 'area').replace('json', 'txt')
        label = shape['label']
        points = shape['points']
        annotation = OrderedDict()
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['area'] = self._get_area(area_path)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # 获取面积
    def _get_area(self, area_path):
        with open(area_path, 'r') as f:
            area = f.readline()
            return float(area)


if __name__ == '__main__':
    labelme_path = "./labelme/"
    saved_coco_path = "./"
    # 创建文件
    if not os.path.exists("{}coco/annotations/".format(saved_coco_path)):
        os.makedirs("{}coco/annotations/".format(saved_coco_path))
    if not os.path.exists("{}coco/images/train2017/".format(saved_coco_path)):
        os.makedirs("{}coco/images/train2017".format(saved_coco_path))
    if not os.path.exists("{}coco/images/val2017/".format(saved_coco_path)):
        os.makedirs("{}coco/images/val2017".format(saved_coco_path))
    if not os.path.exists("{}coco/images/test2017/".format(saved_coco_path)):
        os.makedirs("{}coco/images/test2017".format(saved_coco_path))
    # 数据划分
    train_path = []
    val_path = []
    test_path = []
    translation_path_list = []
    rotation_path_list = []
    for className in list(classname_to_id.keys()):
        # 获取labelme目录下各个类别的原始图片列表
        json_path_list = glob.glob(labelme_path + '/{}_*.json'.format(className))
        # 打乱顺序
        random.shuffle(json_path_list)
        # 取test 112=16x7张
        temp_test = random.sample(json_path_list, 16)
        test_path.extend(temp_test)
        # 去掉已经选出的图片
        for test in temp_test:
            json_path_list.remove(test)
        # train
        train_path.extend(json_path_list)
        # 该类别在每种变换应增的图片数
        class_new_num_floor = math.floor(class_to_num.get(className) / 7)
        class_new_num_ceil = class_new_num_floor + 1
        for augmentation in data_augment:
            augmentation_path_list = glob.glob(labelme_path + '/{}-{}_*.json'.format(augmentation, className))
            random.shuffle(augmentation_path_list)
            # 数据增强得到的新增图片数目是否大于应增数
            if class_new_num_floor >= len(augmentation_path_list):
                temp = augmentation_path_list
            elif className == 'carcinoma_in_situ' and augmentation in data_augment[2:]:
                temp = random.sample(augmentation_path_list, class_new_num_ceil)
            elif className == 'light_dysplastic' and augmentation in data_augment[:1]:
                temp = random.sample(augmentation_path_list, class_new_num_ceil)
            elif className == 'moderate_dysplastic' and augmentation in data_augment[:2]:
                temp = random.sample(augmentation_path_list, class_new_num_ceil)
            elif className == 'normal_columnar' and augmentation in data_augment[:2]:
                temp = random.sample(augmentation_path_list, 214)
            elif className == 'normal_intermediate' and augmentation in data_augment[:2]:
                temp = random.sample(augmentation_path_list, 298)
            elif className == 'normal_superficiel' and augmentation in data_augment[:2]:
                temp = random.sample(augmentation_path_list, 286)
            else:
                temp = random.sample(augmentation_path_list, class_new_num_floor)
            train_path.extend(temp)
    # val
    random.shuffle(train_path)
    val_path.extend(random.sample(train_path, int(0.1 * len(train_path))))
    for val in val_path:
        train_path.remove(val)
    print("train:", len(train_path), 'val:', len(val_path), 'test:', len(test_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '{}coco/annotations/instances_train2017.json'.format(saved_coco_path))

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '{}coco/annotations/instances_val2017.json'.format(saved_coco_path))

    # 把测试集转化为COCO的json格式
    l2c_test = Lableme2CoCo()
    test_instance = l2c_test.to_coco(test_path)
    l2c_test.save_coco_json(test_instance, '{}coco/annotations/instances_test2017.json'.format(saved_coco_path))

    # 复制图像到images目录下
    for file in train_path:
        shutil.copy(file.replace("json", "BMP"), "{}coco/images/train2017/".format(saved_coco_path))
    for file in val_path:
        shutil.copy(file.replace("json", "BMP"), "{}coco/images/val2017/".format(saved_coco_path))
    for file in test_path:
        shutil.copy(file.replace("json", "BMP"), "{}coco/images/test2017/".format(saved_coco_path))
