# Reconstruction-XCO2
OCO-2/3卫星数据及环境变量数据集从文章中链接可下载；所有数据坐标统一为WGS-1984坐标系，利用"resample.py"批量重采样为0.01°的空间分辨率；
使用“clip.py”使得栅格数据集空间范围对齐；
利用“XCO2.py”进行数据的训练
利用“”导入训练后的模型与环境变量数据集进行数据的重建，并可以实现评价指标的计算
The OCO-2 / 3 satellite data and the environmental variable data set can be downloaded from the link in the article ; all data coordinates are unified as the WGS-1984 coordinate system, and the spatial resolution of 0.01 ° is resampled in batches by ' resample.py '. 
Use ' clip.py ' to align the spatial range of the raster data set ; 
using ' XCO2.py ' for data training. 
Using the ', ' import the trained model and the environmental variable data set to reconstruct the data, and can realize the calculation of the evaluation index.
