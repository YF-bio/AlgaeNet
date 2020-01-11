# 头文件引用
import os
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

Root_Dir = '/Users/andrew/Desktop/AC_Proj/'  # 工程根目录
ProcessedImage_Dir = Root_Dir + 'ProcessedImage/'  # 处理后图片目录
UnProcessedImage_Dir = Root_Dir + 'UnprocessedImage/'  # 未处理图片目录


# 数据集分割函数
def Image_split(Image_Dir, Train_Per, Test_Per, Val_Per):
    for Root, Dirs, Files in os.walk(Image_Dir):  # 遍历统计
        for Dirs_Name in Dirs:  # 获取所有藻类细胞的种类
            os.makedirs(Root_Dir + 'Dataset/' + 'Train/' + Dirs_Name + '/')
            os.makedirs(Root_Dir + 'Dataset/' + 'Test/' + Dirs_Name + '/')
            os.makedirs(Root_Dir + 'Dataset/' + 'Val/' + Dirs_Name + '/')
            for root, dirs, files in os.walk(Image_Dir + Dirs_Name):  # 遍历统计
                Image_Counter = 0
                for Files_Name in files:  # 获取所有藻类细胞图片
                    if Files_Name.endswith('jpg'):
                        Image_Counter += 1
                Train_Num = int(Image_Counter * Train_Per)
                Test_Num = int(Image_Counter * Test_Per) + Train_Num
                Val_Num = int(Image_Counter * Val_Per) + Test_Num
                print(Train_Num, Test_Num, Val_Num)

                for Files_Name in files:  # 获取所有藻类细胞图片
                    if Files_Name.endswith('jpg'):
                        print(int(Files_Name.strip('.jpg')))
                        if int(Files_Name.strip('.jpg')) <= Train_Num:
                            # print(Train_Num)
                            shutil.copyfile(Image_Dir + Dirs_Name + '/' + Files_Name, Root_Dir + 'Dataset/' + 'Train/' + Dirs_Name + '/' + Files_Name)
                        elif Train_Num < int(Files_Name.strip('.jpg')) <= Test_Num:
                            # print(Test_Num)
                            shutil.copyfile(Image_Dir + Dirs_Name + '/' + Files_Name, Root_Dir + 'Dataset/' + 'Test/' + Dirs_Name + '/' + Files_Name)
                        elif Test_Num < int(Files_Name.strip('.jpg')) <= Val_Num:
                            # print(Val_Num)
                            shutil.copyfile(Image_Dir + Dirs_Name + '/' + Files_Name, Root_Dir + 'Dataset/' + 'Val/' + Dirs_Name + '/' + Files_Name)


# pytorch 数据集制作函数
def Dataset_Maker(Dataset_Dir, Batch_Size):
    # 数据集转换
    Transform_Train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor()])

    Trainsform_Test = transforms.ToTensor()

    # 训练集数据制作
    Train_Dataset = datasets.ImageFolder(root=Dataset_Dir + 'Train', transform=Transform_Train)

    Train_Dataloader = DataLoader(dataset=Train_Dataset,  # 数据集 --> Train_Dataset
                                  batch_size=Batch_Size,  # 批处理量
                                  shuffle=True)  # 是否随机 --> True

    # 测试集数据制作
    Test_Dataset = datasets.ImageFolder(root=Dataset_Dir + 'Test', transform=Trainsform_Test)

    Test_Dataloader = DataLoader(dataset=Test_Dataset,  # 数据集 --> Test_Dataset
                                 batch_size=Batch_Size,  # 批处理量
                                 shuffle=True)  # 是否随机 --> True

    # 返回数据集数据
    return Train_Dataloader, Test_Dataloader


if __name__ == '__main__':
    Image_split(ProcessedImage_Dir, 0.8, 0.2, 0)    # 划分数据集 Train : Test = 8 : 2

