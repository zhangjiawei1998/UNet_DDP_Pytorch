
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image
from utils.dice_score import dice_loss, dice_coeff
nn.parallel.DataParallel


img = cv2.imread("E:\\dataset\\shein\\20210226_1302120238099230720.jpg") # (W,H)
img = cv2.resize(img, (225,300))
cv2.imwrite("C:\\Users\\NoT-T\\Desktop\\img+.jpg",img)
cv2.waitKey(0)
#---------------------------------------------

# mask = np.array([0,1,2,2,2,2,2,3]).reshape((2,2,2))
# mask1 = np.max(mask)
# mask_area = np.array(mask == 2)
# mask[mask_area] = 0.5 * mask[mask_area] + np.array(100,100)
# print(mask)

#---------------------------------------------

# pre = torch.arange(0,18).view(1,2,3,3)
# mask = torch.arange(0,18).view(1,2,3,3)

# loss1 = dice_coeff(pre[:,0,:,:], mask[:,0,:,:])
# loss2 = dice_coeff(pre[:,1,:,:], mask[:,1,:,:])
# print(loss1, loss2)
# print((loss1+loss2*0.5)/1.5)
# weights = [1,0.5]
# loss =  1 - dice_loss(pre, mask)
# print(loss)

#---------------------------------------------

# epochs = 300
# learning_rate = 0.001

# model = nn.Conv2d(3, 2, kernel_size=3, stride=2)

# optimer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
# scheduler = optim.lr_scheduler.StepLR(optimer, step_size=epochs//3, gamma=0.1)
# last_lr = 1
# for n in range(epochs):
#     if(last_lr != optimer.param_groups[0]['lr']):
#         print(n, ':', optimer.param_groups[0]['lr'])
#     last_lr = optimer.param_groups[0]['lr']
#     scheduler.step()
  
#---------------------------------------------
      
# img = cv2.imread("E:\dataset\out_mask\mask_000001.jpg")  # mat类型 (H, W, C)
# img_array = np.array(img)   # numpy数组 (H, W, C)
# image = Image.open("E:/dataset/img/000001.jpg") # Image类型 (W, H)
# image_array = np.array(image) # numpy数组 (H, W, C)
# print(image.size)


#---------------------------------------------

# GT = torch.tensor([0, 0, 1, 0, 0, 1, 2, 2, 2]).view(3,3).unsqueeze(0)
# print(GT, GT.shape)
# GT = F.one_hot(GT, 3).permute(0,3,1,2)
# print(GT, GT.shape)

#---------------------------------------------

# GT = torch.tensor([0, 0, 1, 0, 0, 1, 2, 2, 2]).view(3,3).unsqueeze(0)

# pre_out = torch.rand((3,3,3)).unsqueeze(0)

# criterion = nn.CrossEntropyLoss()

# loss = criterion(pre_out, GT)

# print(GT,GT.shape, '\n', pre_out, pre_out.shape)
# print(f'loss={loss}')

#---------------------------------------------


# print(torch.__version__)
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

#---------------------------------------------

# class model(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        
#     def forward(self, x):
#         pass
    
# epochs = 20
# net = model(3, 5)

# optimizer = optim.RMSprop(net.parameters(), lr=0.1, alpha=0.9, eps=1e-08)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1, last_epoch=-1)


# for epoch in range(1, epochs):
#     # tarin
#     optimizer.zero_grad()
#     optimizer.step()
#     lr = optimizer.param_groups[0]['lr']
#     print(f'第{epoch}个epoch的学习率{lr}')
#     scheduler.step()














































































