from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.io import imread
import glob
# 打开图像文件
folder_path ='image/GF-2/'
image_files = glob.glob(folder_path + '*2000.png')
print(image_files)
image1 = imread(image_files[0])
image2 = open(image_files[1])
image3 = open(image_files[2])

# gray_image = image1.convert('RGB')
# 打印位深度信息
print(image1.dtype)