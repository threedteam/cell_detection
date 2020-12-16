# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import re
# from PIL import Image


# # 进行ZCA白化和特征标准化
# '''相关参数：
# ZCA白化：zca_whitening=True，
# 特征标准化：samplewise_center=True, samplewise_std_normalization=True，
# rescale：重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
# '''
# datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rescale=None)
#
# fnames = [os.path.join(r'C:\Users\Administrator\Desktop\pic', fname)
#           for fname in os.listdir(r'C:\Users\Administrator\Desktop\pic')]
# # 选择一张图像进行增强
# for img_path in fnames:
#     # 读取图像并调整大小
#     img = load_img(img_path, target_size=None)
#     # 将其转换为形状 (target_size[0], target_size[1], 3) 的 Numpy 数组
#     x = img_to_array(img)
#     # 将其形状改变为 (1, target_size[0], target_size[1], 3)
#     x = x.reshape((1,) + x.shape)
#     x = x.astype('float32')
#     # 计算依赖于数据的变换所需要的统计信息(均值方差等),
#     # 只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数
#     # datagen.fit(x)
#     i = 0
#     for batch in datagen.flow(x, batch_size=1, save_to_dir=r'C:\Users\Administrator\Desktop\test',
#                               save_prefix=((img_path.split('\\')[-1]).split('.')[0]),
#                               save_format=((img_path.split('\\')[-1]).split('.')[1]), shuffle=False):
#         i += 1
#         if i == 1:
#             break

# 图像重命名
pics = os.listdir(r'C:\Users\Administrator\Desktop\test')
for pic in pics:
    new_name = 'feature_standardization-' + re.split(r'_0_\d{2,4}', pic)[0]
    print(new_name)
    os.rename(r'C:\Users\Administrator\Desktop\test/' + pic, r'C:\Users\Administrator\Desktop\test/' + new_name + '.BMP')


# # 进行水平、竖直镜像
# fnames = [os.path.join('./pictures', fname)
#           for fname in os.listdir('./pictures')]
# for img_path in fnames:
#     img = Image.open(img_path)
#     out = img.transpose(Image.FLIP_LEFT_RIGHT)
#     # out = img.transpose(Image.FLIP_TOP_BOTTOM)
#     out.save('./水平镜像/' + 'horizontal_flip-' + img_path.split('\\')[-1])
#     # out.save('./垂直镜像/' + 'vertical_flip-' + img_path.split('\\')[-1])


# # 细胞类别
# classes = ['carcinoma_in_situ', 'light_dysplastic',
#            'normal_columnar', 'normal_intermediate',
#            'normal_superficiel', 'severe_dysplastic',
#            'moderate_dysplastic']
# # 文件路径
# path = './New_database_pictures/'
# for i in classes:
#     file_dir = path + i
#     class_name = i
#     files = os.listdir(file_dir)
#     # 对单个图操作
#     for file in files:
#         flag = file.rsplit("-", 1)[-1]
#         if flag == 'd.bmp':
#             fullPath = file_dir + '/' + file
#             img = Image.open(fullPath)
#             # out = img.transpose(Image.FLIP_LEFT_RIGHT)
#             out = img.transpose(Image.FLIP_TOP_BOTTOM)
#             writeFolderPath = './染色图垂直镜像/' + class_name
#             isExists = os.path.exists(writeFolderPath)
#             if not isExists:
#                 os.makedirs(writeFolderPath)
#             writeFilePath = './染色图垂直镜像/' + class_name + '/' + \
#                             file.split('\\')[-1]
#             out.save(writeFilePath)
#