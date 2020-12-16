from pylab import *
import os
import json
import cv2


# 半自动标注图像，并生成可供labelme接口解析的json类型的文件
def dict_other_json(shapes, img_path, img_json, img_height, img_width):
    return {"version": "3.16.7", "flags": {}, 'shapes': shapes, "lineColor": [0, 255, 0, 128],
            'fillColor': [255, 0, 0, 128], "imagePath": img_path, "imageData": img_json,
            'imageHeight': img_height, 'imageWidth': img_width}


def dict_shapes(label, points):
    return {'label': label, 'line_color': None, 'fill_color': None,
            'points': points, 'shape_type': 'polygon', 'flags': {}}


def fusion(file_dir, imgs_path, img_jsons_path, fusion_path, class_name):
    """将坐标数据与对应的图像进行融合，生成可以替代labelme生成的json文件"""
    files = os.listdir(file_dir)
    for file in files:
        if file == 'desktop.ini':
            continue
        shapes = []
        points = []
        # print(file)
        # print(file.split(".")[0])
        img_json = json.load(open(img_jsons_path+file.split(".")[0]+'.json'))
        img_name = file.split(".")[0] + '.BMP'
        img_fullPath = imgs_path + img_name
        img_size = cv2.imread(img_fullPath).shape
        img_height = img_size[0]
        img_width = img_size[1]
        full_path = file_dir + '/' + file
        with open(full_path, 'r') as f:
            for i in f.readlines():
                for j in eval(i):
                    point_temp = []
                    point_temp.append(float(j[0]))
                    point_temp.append(float(j[1]))
                    points.append(point_temp)
        shapes.append(dict_shapes(class_name, points))
        data = dict_other_json(shapes, img_name, img_json, img_height, img_width)
        # print(data)
        # 写入json文件
        if not os.path.exists(fusion_path):
            os.mkdir(fusion_path)
        json_file = fusion_path + file.split(".")[0] + '.json'
        json.dump(data, open(json_file, 'w'))


if __name__ == "__main__":
    points_path = "./points/"  # 边界坐标数据文件路径
    classes = ['carcinoma_in_situ', 'light_dysplastic',
               'normal_columnar', 'normal_intermediate',
               'normal_superficiel', 'severe_dysplastic',
               'moderate_dysplastic']
    img_jsons_path = "./pictures_json/"
    imgs_path = "./pictures/"
    fusion_path = "./result_json/"
    for i in classes:
        points_fullPath = points_path + i + '/'
        fusion(points_fullPath, imgs_path, img_jsons_path, fusion_path, i)
