import cv2
import os
import heapq
import numpy as np


# import picture
def findconts(file_dir, class_name):
    '''
                Find the coordinates of the target contour for each image
    '''
    files = os.listdir(file_dir)
    for file in files:
        flag = file.rsplit(".", 1)[-1]

        # Operation on processed cell images
        if flag == 'jpg':
            fullPath = file_dir + '/' + file
            src = cv2.imread(fullPath)
            size = src.shape
            height = size[0]
            width = size[1]

            #  Expand the image by 10 pixels to the left,right,up and down
            src_padding = cv2.copyMakeBorder(src, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            # Convert image to grayscale image
            gray = cv2.cvtColor(src_padding, cv2.COLOR_BGR2HSV)

            # Binarization
            lower_hsv = np.array([100, 43, 46])  # Set blue lower limit
            upper_hsv = np.array([124, 255, 255])  # Set blue upper limit
            mask = cv2.inRange(gray, lowerb=lower_hsv, upperb=upper_hsv)  # According to the upper and lower limits, the target image is transformed into binary image

            # Find Contours, use cv2.CHAIN_APPROX_NONE to find all contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # because we use cv2.CHAIN_APPROX_NONE to find all contours,the largest contours is the boundary Contour of image,
            # now we choose the second largest contours
            contours_each_length = []
            for contour in contours:
                contourPoint = contour.tolist()
                lenth = len(contourPoint)
                contours_each_length.append(lenth)
            position = list(map(contours_each_length.index, heapq.nlargest(2, contours_each_length)))[0]
            contour_list = contours[position].tolist()

            # Calculate the area of the contours
            area = cv2.contourArea(contours[position])

            # Save the area of the target contours for each image
            AreaFolder = './Area/' + str(file_dir.split('/')[1])+ 'Area/'
            writeAreaFolderPath = AreaFolder + class_name
            if not os.path.exists(AreaFolder):
                os.mkdir(AreaFolder)
            if not os.path.exists(writeAreaFolderPath):
                os.mkdir(writeAreaFolderPath)
            writeAreaFilePath = writeAreaFolderPath + '/' + file.rsplit(".", 1)[0] + '.txt'
            print(writeAreaFilePath)
            fA = open(writeAreaFilePath, 'w')
            fA.write(str(area))
            fA.close()

            # The contour coordinates minus the pixels that extend outward
            contours_length = len(contour_list)
            list_contours = []
            for i in range(0, contours_length):
                origin_position = []
                origin_position_x = contour_list[i][0][0] - 10
                origin_position_y = contour_list[i][0][1] - 10
                if (origin_position_x < 0):
                    origin_position_x = 0
                if (origin_position_y < 0):
                    origin_position_y = 0
                if (origin_position_x > width):
                    origin_position_x = width
                if (origin_position_y > height):
                    origin_position_y = height
                origin_position.append(origin_position_x)
                origin_position.append(origin_position_y)

                list_contours.append(origin_position)


            # Saves the coordinates of the target contour for each image
            ContoursFolder = './Contours/' + str(file_dir.split('/')[1])+ 'Contours/'
            writeContoursFolderPath = ContoursFolder + class_name
            if not os.path.exists(ContoursFolder):
                os.mkdir(ContoursFolder)
            if not os.path.exists(writeContoursFolderPath):
                os.mkdir(writeContoursFolderPath)
            writeContoursFilePath = writeContoursFolderPath + '/' + file.rsplit(".", 1)[0] + '.txt'
            print(writeContoursFilePath)
            fC = open(writeContoursFilePath,'w')
            fC.write(str(list_contours))
            fC.close()


# Cell type
classes = ['carcinoma_in_situ', 'light_dysplastic',
           'normal_columnar', 'normal_intermediate',
           'normal_superficiel', 'severe_dysplastic',
           'moderate_dysplastic']

if not os.path.exists('./Area'):
    os.mkdir('./Area')
if not os.path.exists('./Contours'):
    os.mkdir('./Contours')

path = ['./rotationImg2/', './pingyiImg2/', './scaleImg2/']
for m in path:
    for i in classes:
        findconts(m+i , i)

