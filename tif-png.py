import cv2
import numpy as np
from skimage.io import imread
import glob

folder_path = 'image/huanghefenlei/'
save_path = 'image/huanghefenlei(png)/'
image_files = glob.glob(folder_path + '*.tif')
image_files2 = glob.glob(save_path + '*2000.png')

print(image_files)
# image1 = imread(image_files[0])
# image2 = imread(image_files[1])
# image3 = imread(image_files[2])
# print(image1.shape)
# print(image2.shape)
# print(image2.shape)
#tif转换png
for i in range(3):
    image = imread(image_files[i])
    cv2.imwrite(save_path + str(i + 1) + "2000.png", image)
    print("转换成功")