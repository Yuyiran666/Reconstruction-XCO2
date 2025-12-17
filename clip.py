# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:19 2022
@author: fkxxgis
"""
import arcpy
tif_file_path = r""#输入路径
result_file_path = r""#输出路径
snap_file_name = r""#裁剪目标范围
arcpy.env.workspace = tif_file_path
arcpy.env.snapRaster = snap_file_name
tif_file_list = arcpy.ListRasters("*", "tif")
for tif_file in tif_file_list:
    print(tif_file)
    key_name = tif_file.split(".tif")[0] + ".tif"
    # key_name = "NO2.tif" + tif_file.split(".tif")[0]
    arcpy.Clip_management(tif_file,
                          "#",
                          result_file_path + key_name,
                          snap_file_name,
                          "-3.402823e+38",
                          "#",
                          "MAINTAIN_EXTENT")