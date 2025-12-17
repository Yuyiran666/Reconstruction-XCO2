import numpy as np
from osgeo import gdal
import pickle
import glob

# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
    Tif_height = dataset.RasterYSize  # 栅格矩阵的行数
    Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    Tif_proj = dataset.GetProjection()  # 获取投影信息
    Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)

    if len(Landset_data.shape) == 3:
        data = np.zeros((Landset_data.shape[0], Landset_data.shape[1] * Landset_data.shape[2]))
        for i in range(Landset_data.shape[0]):
            data[i] = Landset_data[i].flatten()
    else:
        data = Landset_data.flatten()
        data=np.expand_dims(data,0)
    return data,Tif_geotrans,Tif_proj

# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape

    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)

    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
#设置路径
folder_path = r''
# 模型所在路径
modelpath = r""
# 环境变量栅格数据集所在文件夹
tif_files = glob.glob(folder_path + '/*.tif', recursive=True)
# print(tif_files)
# 保存路径
SavePath = r""
file = open(modelpath, "rb")
model = pickle.load(file)
file.close()

img_list = []
#遍历tif文件，并将完整文件名添加进列表
for i in range(len(tif_files)):
    if i ==0:
       dataset,Tif_geotrans,Tif_proj = readTif(tif_files[i])
       img_list.append(dataset)
    else:
        dataset, _, _ = readTif(tif_files[i])
        img_list.append(dataset)
#拼接不同tif文件进同一个数组
data = np.concatenate(img_list, axis=0)
data = data.swapaxes(0, 1)
pred = TAF_model.predict(data)
##  同样地，我们对预测好的数据调整为我们图像的格式
print(pred.shape)
# 栅格行列对齐
pred = pred.reshape(3556,6144)
print(pred)
# pred = pred.astype(np.uint8)
# # # #
# #  将结果写到tif图像里
writeTiff(pred, Tif_geotrans, Tif_proj, SavePath)