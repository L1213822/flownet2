import torch
import numpy as np
import argparse
import time
from models import FlowNet2  # the path is depended on where you create this module

from utils.flow_utils import flow2img
import matplotlib.pyplot as plt
import cv2
image_path = 'image/huanghefenlei(png)/'
save_path = 'image/huanghefenlei/result/'  #结果保存位置
image_name = '20' #图像名称，更改序号来实现预测不同的图像对
model_path = "FlowNet2_checkpoint.pth.tar"  # 模型位置

'''
crop_size 需要根据图像的实际尺寸更改，且是32的倍数
'''
crop_size = (512, 448)
prvs = cv2.imread(image_path + image_name+'00(gengdi).png')
next = cv2.imread(image_path + image_name+'20(gengdi).png')
#pim1 = prvs
#pim2 = next
pim1 = cv2.resize(prvs, crop_size, interpolation = cv2.INTER_CUBIC)
pim2 = cv2.resize(next, crop_size, interpolation = cv2.INTER_CUBIC)

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)

args = parser.parse_args()
net = FlowNet2(args).cuda()
dict = torch.load(model_path)
net.load_state_dict(dict["state_dict"])

images = [pim1, pim2]
images = np.array(images).transpose(3, 0, 1, 2)
im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

start = time.time()
result = net(im).squeeze()
end = time.time()
print(end-start)
data = result.data.cpu().numpy().transpose(1, 2, 0)
img = flow2img(data)
cv2.imwrite(save_path + image_name[:1] +'2002.png',img)
print(result.shape)

