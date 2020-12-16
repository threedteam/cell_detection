from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import os
from PIL import Image
# import cv2

def data_augmentation(classname,file):
    imagegen = ImageDataGenerator(
        width_shift_range=10,
        height_shift_range=10,
        # rotation_range=180,
        fill_mode='constant'
    )
    # maskgen = ImageDataGenerator(
    #     width_shift_range=20,
    #     fill_mode='constant'
    # )

    img_path = "./paddingImg_origin/"  + classname + "/" + file
    im = Image.open(img_path)
    img = load_img(img_path)

    # scale the image
    # img = load_img(img_path,target_size=(int(im.size[1]*1.3),int(im.size[0]*1.3)))
    
    x = img_to_array(img, data_format="channels_last") 
    x = x.reshape((1,) + x.shape)  

    filename = file.rsplit(".", 1)[0]
    mask_path = "./paddingImg_after/" + classname + "/" + filename + "-d.bmp"
    ma = Image.open(mask_path)
    mask = load_img(mask_path)

    # scale the image
    # mask = load_img(mask_path,target_size=(int(ma.size[1]*1.3),int(ma.size[0]*1.3)))
    
    y = img_to_array(mask, data_format="channels_last")  
    y = y.reshape((1,) + y.shape)  

    # folderName1 = './pingyiImg/'  
    # folderName2 = './pingyiImg2/'
    # folderName1 = './rotationImg/'  
    # folderName2 = './rotationImg2/'
    folderName1 = './scaleImg/'  
    folderName2 = './scaleImg2/'
    if not os.path.exists(folderName1):
        os.mkdir(folderName1)
    if not os.path.exists(folderName2):
        os.mkdir(folderName2)

    imag_save = folderName1 + classname + "/"
    mask_save = folderName2 + classname + "/"
    if not os.path.exists(imag_save):
        os.mkdir(imag_save)
    if not os.path.exists(mask_save):
        os.mkdir(mask_save)

    i = 1
    for batch in imagegen.flow(x,
                              batch_size=1,
                              seed=1,
                              save_to_dir=imag_save,  
                              save_prefix = filename,
                              save_format='jpg'):
        i += 1
        if i > 5:  
            break  

    i = 1
    for batch in imagegen.flow(y,
                              batch_size=1,
                              seed=1,
                              save_to_dir=mask_save,  
                              save_prefix=filename,
                              save_format='jpg'):
        i += 1
        if i > 5:  
            break 

classes = ['carcinoma_in_situ' , 'light_dysplastic',
           'normal_columnar', 'normal_intermediate',
           'normal_superficiel', 'severe_dysplastic',
           'moderate_dysplastic']




for i in classes:
    path = "./paddingImg_origin/" + i + '/'
    files = os.listdir(path)
    for file in files:
        data_augmentation(i,file)