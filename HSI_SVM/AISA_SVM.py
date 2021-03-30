# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:05:45 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:56:08 2021

@author: wx
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:29:01 2021

@author: wx
"""

# pip install spectral

# 程序说明：此程序用于基于珠海一号的SVM水体提取的方法-2021.01.18

# 1、读取excel表，获取数组大小，创建数组，并将数据存到相应的数组中
# 1.1 读取excel表，获取数组大小
import xlrd
import os
import numpy as np
from spectral import *
import time

time_start=time.time()

traindata = xlrd.open_workbook(r'F:\wx\JXSVM\traindata.xlsx')
nowatersheet = traindata.sheet_by_name('nowater')
watersheet = traindata.sheet_by_name('water')
print(watersheet.name,watersheet.nrows,watersheet.ncols)
print(nowatersheet.name,nowatersheet.nrows,nowatersheet.ncols)
# 1.2 创建数组
NoWaterArray = np.zeros([nowatersheet.nrows,nowatersheet.ncols])
WaterArray = np.zeros([watersheet.nrows,watersheet.ncols])
# print(NoWaterArray)
# print(WaterArray)
# 1.3 将数据存到相应的数组中
for i in range(0,nowatersheet.nrows):
    NoWaterArray[i] = nowatersheet.row_values(i)
for i in range(0,watersheet.nrows):
    WaterArray[i] = watersheet.row_values(i)
# print(NoWaterArray)
# print(WaterArray)

# 2、进行数据归一化，/10000；

# 3、添加相对应的Label标签
WLabel = np.zeros([len(WaterArray),1])
NLabel = np.zeros([len(NoWaterArray),1])

for i in range(0,len(WLabel)):
    WLabel[i] = 1

train_data =  np.insert(WaterArray,0,NoWaterArray,axis=0)
train_label =  np.insert(WLabel,0,NLabel,axis=0)

# 4、进行交叉验证，并利用最适合的参数去训练样本
# 4.1 交叉验证结果如下：

# 4.2 使用交叉验证的参数去训练训练集
from sklearn.svm import SVC
classifier = SVC(C=1,kernel='linear',decision_function_shape='ovo')
# classifier = SVC(C=,kernel='rbf',gamma=0.01,decision_function_shape='ovo')
classifier.fit(train_data,train_label.ravel())
print("训练集：",classifier.score(train_data,train_label))
# 训练集： 0.9933333333333333

######################################################################释放内存
import gc

del NoWaterArray,WaterArray,WLabel,NLabel
gc.collect()

#############################################################################


# 5、用训练好的模型去识别提取水体
# 5.1 读取识别影像数据
Datapath = "F:/wx/JiaXing/mosaic.hdr" 

# 	Data Source:   'F:/wx/JiaXing/mosaic.dat'
# 	# Rows:           6718
# 	# Samples:       13708
# 	# Bands:            66
# 	Interleave:        BSQ
# 	Quantization:  32 bits
# 	Data format:   float32

Datalib = envi.open(Datapath)
DataRows = Datalib.nrows
DataCols = Datalib.ncols
DataBands = Datalib.nbands
print(Datalib)

# 前六排的存储位置
PixelTest = np.zeros([(1000*DataCols),DataBands])

# 01
for i in range(0,1000):
    for j in range(0,DataCols):
        PixelTest[(i*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict01 = classifier.predict(PixelTest)

######################################################################释放内存

del PixelTest
gc.collect()

#############################################################################

PixelTest = np.zeros([(1000*DataCols),DataBands])

# 02
for i in range(1000,2000):
    for j in range(0,DataCols):
        PixelTest[((i-1000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict02 = classifier.predict(PixelTest)

Y_temp1 = np.hstack((Y_predict01,Y_predict02))
######################################################################释放内存

del Y_predict01,Y_predict02,PixelTest
gc.collect()

#############################################################################

PixelTest = np.zeros([(1000*DataCols),DataBands])

# 03
for i in range(2000,3000):
    for j in range(0,DataCols):
        PixelTest[((i-2000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict03 = classifier.predict(PixelTest)

######################################################################释放内存

del PixelTest
gc.collect()

#############################################################################

PixelTest = np.zeros([(1000*DataCols),DataBands])

# 04
for i in range(3000,4000):
    for j in range(0,DataCols):
        PixelTest[((i-3000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict04 = classifier.predict(PixelTest)

######################################################################释放内存

del PixelTest
gc.collect()

#############################################################################

Y_temp2 = np.hstack((Y_predict03,Y_predict04))
Y_temp4 = np.hstack((Y_temp1,Y_temp2))
######################################################################释放内存

del Y_predict03,Y_predict04,Y_temp1,Y_temp2
gc.collect()

#############################################################################

PixelTest = np.zeros([(1000*DataCols),DataBands])

# 05
for i in range(4000,5000):
    for j in range(0,DataCols):
        PixelTest[((i-4000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict05 = classifier.predict(PixelTest)

######################################################################释放内存

del PixelTest
gc.collect()

#############################################################################

PixelTest = np.zeros([(1000*DataCols),DataBands])

# 06
for i in range(5000,6000):
    for j in range(0,DataCols):
        PixelTest[((i-5000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict06 = classifier.predict(PixelTest)

Y_temp3 = np.hstack((Y_predict05,Y_predict06))
######################################################################释放内存

del Y_predict05,Y_predict06,PixelTest
gc.collect()

#############################################################################

Pixellast = np.zeros([(718*DataCols),DataBands])
# 07
for i in range(6000,6718):
    for j in range(0,DataCols):
        Pixellast[((i-6000)*DataCols+j)] = Datalib.read_pixel(i, j)
    print(i)
Y_predict07 = classifier.predict(Pixellast)

######################################################################释放内存

del Pixellast
gc.collect()

#############################################################################
# 5.2 SVM predict
# np.append(Y_predict,Y_predict01)
# np.append(Y_predict,Y_predict02)
# np.append(Y_predict,Y_predict03)

Y_temp5 = np.hstack((Y_temp4,Y_temp3))
Y_predict = np.hstack((Y_temp5,Y_predict07))

######################################################################释放内存

del Y_predict07
del Y_temp3,Y_temp4,Y_temp5
gc.collect()

#############################################################################


# watersum = 0
# for i in range(0,len(Y_predict)):
#     if Y_predict[i] == 1:
#         watersum = watersum + 1
# print("watersum = ",watersum)

# 5.3 返回提取的像素位置
Location = Y_predict.reshape(DataRows,DataCols)

# 5.4生成二值结果图
from PIL import Image
im = Image.fromarray(Location)
# 得到svm_classic结果
im.save("svm_classic.tif")

time_end=time.time()
print('time cost',time_end-time_start,'s')
# # time cost 6235.493759632111 s

