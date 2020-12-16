import cv2
import os
import heapq


# import picture
def findconts(file_dir, class_name):
    '''
               Let the cells pad out to facilitate the subsequent data enhancement work
    '''
    files = os.listdir(file_dir)
    for file in files:

        # Operation on processed cell images
        fullPath = file_dir + '/' + file
        src = cv2.imread(fullPath)
        size = src.shape
        height = size[0]
        width = size[1]
        #  Expand the image by 10 pixels to the left,right,up and down
        src_padding = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # folderName = './paddingImg_origin/'
        folderName = './paddingImg_after/'  
        if not os.path.exists(folderName):
            os.mkdir(folderName)
        writeFolder = folderName + class_name
        isExists = os.path.exists(writeFolder)
        if not isExists:
            os.mkdir(writeFolder)
        writePath = folderName + class_name +"/" +file
        cv2.imwrite(writePath,src_padding)



# Cell type
classes = ['carcinoma_in_situ', 'light_dysplastic',
           'normal_columnar', 'normal_intermediate',
           'normal_superficiel', 'severe_dysplastic',
           'moderate_dysplastic']

# File path
# path = '../New_database_pictures_origin/'
path = '../New_database_pictures_after/'
for i in classes:
    findconts(path + i, i)
