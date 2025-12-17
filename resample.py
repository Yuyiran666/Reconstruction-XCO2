# encoding:utf-8
import glob
import os
import arcpy

if __name__ == "__main__":

    file = r""# 输入路径
    out = r""# 输出路径
    file = unicode(file, "utf-8")
    out = unicode(out, "utf-8")
    os.chdir(file)
    # names = arcpy.ListRasters("*", "tif")
    # names = glob.glob(os.path.join(file, "*.tif"))
    names = os.listdir(file)

    for name in names:
        print(name)
        filepath = file + '\\' + name
        output = out + '\\' + name
        arcpy.Resample_management(filepath, output, "0.01", "BILINEAR")  # 这个是在地理坐标的基础上进行的重采样
    # 删除多余的文件
    for file_i in glob.glob(os.path.join(out, '*.xml')):
        os.remove(file_i)
    for file_i in glob.glob(os.path.join(out, '*.tfw')):
        os.remove(file_i)
    for file_i in glob.glob(os.path.join(out, '*.ovr')):
        os.remove(file_i)

