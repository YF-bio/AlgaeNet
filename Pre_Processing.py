# 头文件引用
import cv2
import numpy as np
import os

# 路径声明
Root_Dir = '/Users/andrew/Desktop/AC_Proj/'  # 工程根目录
ProcessedImage_Dir = Root_Dir + 'ProcessedImage/'  # 处理后图片目录
UnProcessedImage_Dir = Root_Dir + 'UnprocessedImage/'  # 未处理图片目录
Single_Target_Dir = Root_Dir + 'Single_Target/'


# 训练数据处理函数
def Train_Data_Processing(Image_Num, Image_Class):
    # 图像一级处理（原图 ---> 二值化图像）
    img = cv2.imread(UnProcessedImage_Dir + Image_Class + '/' + Image_Num + '.tif')  # 读取图片
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转换至灰度图
    ret, BinaryImage = cv2.threshold(GrayImage, 200, 255, cv2.THRESH_BINARY_INV)  # 对灰度图做二值化

    # 图像二级处理（二值化图像 ---> 单细胞图像提取）
    # 二值化图像孔洞填充
    Binary_FloodFill = BinaryImage.copy()
    h, w = BinaryImage.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(Binary_FloodFill, mask, (0, 0), 255)
    Binary_FloodFill_Inv = cv2.bitwise_not(Binary_FloodFill)
    Binary_FloodFill_Out = BinaryImage | Binary_FloodFill_Inv

    # 单细胞轮廓提取
    contours, binary, = cv2.findContours(Binary_FloodFill_Out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 单细胞图像提取
    for i in range(0, len(contours) - 1):
        # 提取藻类细胞边界的坐标值，(x, y) 为左上点坐标　(x + w, Y + h) 为右下点坐标
        x, y, w, h = cv2.boundingRect(contours[i])
        # 获取单个细胞图像
        Single_Target = img[y - 2:(y + h + 2), x - 2:(x + w + 2)]

        # 保存图片
        cv2.imwrite(Single_Target_Dir + Image_Class + '/' + Image_Num + '_' + str(i) + '.jpg', Single_Target)


def Mix_Image(Image_Num, Image_Class):
    # 图像背景读取
    BackGround = cv2.imread(UnProcessedImage_Dir + 'BackGround.jpg')

    Single_Target = cv2.imread(Single_Target_Dir + Image_Class + '/' + Image_Num + '.jpg')  # 读取图片

    Target_Background = BackGround.copy()
    Target_Height, Target_Width = Single_Target.shape[:2]

    # 粘贴坐标设定
    W_Beg = int(20 - Target_Width / 2)
    W_End = W_Beg + Target_Width
    H_beg = int(20 - Target_Height / 2)
    H_End = H_beg + Target_Height

    # 粘贴
    Target_Background[H_beg:H_End, W_Beg:W_End, :] = Single_Target

    # 保存图片
    cv2.imwrite(ProcessedImage_Dir + Image_Class + '/' + str(Image_Num) + '.jpg', Target_Background)


if __name__ == '__main__':
    # 获取显微镜图像数量
    path = Single_Target_Dir    # 获取目标路径
    for Root, Dirs, Files in os.walk(path):  # 遍历统计
        for Dirs_Name in Dirs:  # 获取所有藻类细胞的种类
            for root, dirs, files in os.walk(path + Dirs_Name):  # 遍历统计
                for Files_Name in files:  # 获取所有藻类细胞图片
                    if Files_Name.endswith('jpg'):
                        Mix_Image(Files_Name.strip('.jpg'), Dirs_Name)
    #                     Train_Data_Processing(Files_Name.strip('.tif'), Dirs_Name)
