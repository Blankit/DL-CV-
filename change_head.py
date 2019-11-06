# coding:utf-8
'''
证件照换背景
'''
'''
1. 分割出背景区域的掩膜（将图像转成hsv格式，通过cv2.inrange()选出特定颜色区域的背景）
背景区域的掩膜：背景是白色，头像部分是黑色的
2. 对背景区域的掩膜取反，得到反掩膜(外黑里白)
3. 画出与证件照相同大小的背景
4. 新背景：新背景与掩膜按位与
5. 分割头像区域。反掩膜与原图按位与
6. 目标证件照：新背景与头像区域按位或。

'''

import cv2
import numpy as np
img = cv2.imread(r'D:\MyData\zengxf\Desktop\temp\shu.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
# LowerBlue = np.array([100, 100, 50])
#
# UpperBlue = np.array([130, 255, 255])
LowerBlue = np.array([20, 120, 100])#BGR
UpperBlue = np.array([130, 255, 200])
mask = cv2.inRange(hsv, LowerBlue, UpperBlue)
mask_not = cv2.bitwise_not(mask)# 外黑里白
# 画出蓝色背景
print(img.shape)
blank = np.zeros(img.shape,dtype = np.uint8)
color = (218,143,3)# BGR
# color = (255,0,0)
background = cv2.rectangle(blank,(0,0),(425,602),color = color,thickness=-1)
# show(img_rectangle)
background = cv2.bitwise_and(background,background,mask=mask)

# 抠出头像
head = cv2.bitwise_and(img,img,mask=mask_not)
# 合并
combination = cv2.bitwise_or(background,head)
# cv2.imshow('background',background)
# cv2.imshow('head',head)
# cv2.imshow('shu',combination)
cv2.imwrite('./shu.jpeg',combination)
# cv2.imshow('shu',mask)
# k = cv2.waitKey(0)  # 无限等待一个键击，将此键击存在k变量中
# if k == 27:  # 27代表esc，可以查看ascii码表
#     cv2.destroyAllWindows()  # 退出窗口