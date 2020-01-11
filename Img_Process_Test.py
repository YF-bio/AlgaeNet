# 头文件引用
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 目录声明
Root_Dir = '/Users/andrew/Desktop/AC_Proj/'  # 工程根目录
UnProcessedImage_Dir = Root_Dir + 'UnprocessedImage/'  # 未处理图片目录
TestImg_Save_Dir = Root_Dir + 'Others/Test_Img/'  # 测试图片存储目录

# 开始进行测试
# 读取图片
Image_Test = cv2.imread(UnProcessedImage_Dir + 'Chlorella/1.tif')
# 源图像转灰度图
GrayImage = cv2.cvtColor(Image_Test, cv2.COLOR_BGR2GRAY)
cv2.imwrite(TestImg_Save_Dir + 'GrayImage.jpg', GrayImage)

# 对灰度图做二值化
ret, BinaryImage = cv2.threshold(GrayImage, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite(TestImg_Save_Dir + 'BinaryImage.jpg', BinaryImage)

# 二值化图像孔洞填充
Binary_FloodFill = BinaryImage.copy()
h, w = BinaryImage.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(Binary_FloodFill, mask, (0, 0), 255)
Binary_FloodFill_Inv = cv2.bitwise_not(Binary_FloodFill)
Binary_FloodFill_Out = BinaryImage | Binary_FloodFill_Inv
cv2.imwrite(TestImg_Save_Dir + 'FloodFillImage.jpg', Binary_FloodFill_Out)

# 轮廓提取
_, contours, hierarchy = cv2.findContours(Binary_FloodFill_Out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(Image_Test, contours, -1, (0, 0, 255), 1)
cv2.imwrite(TestImg_Save_Dir + 'ContoursImage.jpg', Image_Test)


# 图片测试
# cv2.imshow('Test', GrayImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
