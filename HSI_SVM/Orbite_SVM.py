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
# 程序说明：此程序用于基于珠海一号的SVM水体提取的方法-2021.01.18

# SVM方法的步骤：
# 1.Transform data to the format of an SVM package
# 读取excel表中的数据，分别放到指定的数组中；
# 2.Conduct simple scaling on the data
# 进行数据的归一化；
# 3.Consider the RBF kernel
# 考虑使用RBF和Linear核函数；
# 4.Use cross-validation to find the best parameter
# 使用交叉验证，选择精度较高的系数；
# 5.Use the best parameter to train the whole training set
# 使用最适合的系数去训练样本数据；
# 6.Test
# 使用最适合的参数去识别图像

# 1、读取excel表，获取数组大小，创建数组，并将数据存到相应的数组中
# 1.1 读取excel表，获取数组大小
import xlrd
import os
import numpy as np
from spectral import *
import time

time_start=time.time()

traindata = xlrd.open_workbook(r'E:\WaterExtract\Orbite\SVM\traindata 2.0.xlsx')
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
for i in range(0,len(NoWaterArray)):
    for j in range(0,len(NoWaterArray[0])):
        NoWaterArray[i][j] = NoWaterArray[i][j]/10000
for i in range(0,len(WaterArray)):
    for j in range(0,len(WaterArray[0])):
        WaterArray[i][j] = WaterArray[i][j]/10000
# print(NoWaterArray)
# print(WaterArray)

# for i in range(0,nowatersheet.nrows):
#     NoWaterArray[i] = nowatersheet.row_values(i)
# for i in range(0,watersheet.nrows):
#     WaterArray[i] = watersheet.row_values(i)

# 3、添加相对应的Label标签
WLabel = np.zeros([len(WaterArray),1])
NLabel = np.zeros([len(NoWaterArray),1])

for i in range(0,len(WLabel)):
    WLabel[i] = 1

train_data =  np.insert(WaterArray,0,NoWaterArray,axis=0)
train_label =  np.insert(WLabel,0,NLabel,axis=0)

# 4、进行交叉验证，并利用最适合的参数去训练样本
# 4.1 交叉验证结果如下：
# 0.012 (+/-0.000) for {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
# 0.859 (+/-0.102) for {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.974 (+/-0.012) for {'C': 1, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 10, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 100, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 1000, 'kernel': 'linear'}


# 4.2 使用交叉验证的参数去训练训练集
from sklearn.svm import SVC
classifier = SVC(C=1,kernel='linear',decision_function_shape='ovo')
# classifier = SVC(C=,kernel='rbf',gamma=0.01,decision_function_shape='ovo')
classifier.fit(train_data,train_label.ravel())
print("训练集：",classifier.score(train_data,train_label))
# 训练集： 0.9116666666666666

# 5、用训练好的模型去识别提取水体
# 5.1 读取识别影像数据
Datapath = "E:/WaterExtract/Orbite/PreproData/Flaashresult.hdr" 

Datalib = envi.open(Datapath)
DataRows = Datalib.nrows
DataCols = Datalib.ncols
DataBands = Datalib.nbands
print(Datalib)

PixelTest = np.zeros([(DataRows*DataCols),DataBands])
for i in range(0,DataRows):
    for j in range(0,DataCols):
        PixelTest[(i*(DataCols)+j)] = Datalib.read_pixel(i, j)
# print(PixelTest[4])

# 5.2 SVM predict
Y_predict = classifier.predict(PixelTest)

watersum = 0
for i in range(0,len(Y_predict)):
    if Y_predict[i] == 1:
        watersum = watersum + 1
print("watersum = ",watersum)

# 5.3 返回提取的像素位置
Location = Y_predict.reshape(DataRows,DataCols)

# 5.4生成二值结果图
from PIL import Image
im = Image.fromarray(Location)
# 得到svm_classic结果
im.save("svm_classic.tif")

time_end=time.time()
print('time cost',time_end-time_start,'s')
# # time cost 1304.0011003017426 s

#———————————————————————————————————————
# 附件
# 附件1 交叉验证代码
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.svm import SVC

# digits = datasets.load_digits()

# n_samples = len(digits.images)
# X = digits.images.reshape((n_samples, -1))
# y = digits.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=0)

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-2],
#                       'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#       # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                         scoring='%s_macro' % score)
#     # 用训练集训练这个学习器 clf
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
    
#     # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
#     print(clf.best_params_)
    
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
    
#     # 看一下具体的参数间不同数值的组合后得到的分数是多少
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
              
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
    
#     # 打印在测试集上的预测结果与真实值的分数
#     print(classification_report(y_true, y_pred))
    
#     print()

# 附件2 交叉验证结果
# Best parameters set found on development set:

# {'C': 1, 'kernel': 'linear'}

# Grid scores on development set:

# 0.012 (+/-0.000) for {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
# 0.859 (+/-0.102) for {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.012 (+/-0.000) for {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}
# 0.921 (+/-0.007) for {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.974 (+/-0.012) for {'C': 1, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 10, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 100, 'kernel': 'linear'}
# 0.974 (+/-0.012) for {'C': 1000, 'kernel': 'linear'}

# Detailed classification report:

# The model is trained on the full development set.
# The scores are computed on the full evaluation set.

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        89
#            1       0.96      0.96      0.96        90
#            2       0.98      0.99      0.98        92
#            3       0.96      0.99      0.97        93
#            4       0.99      1.00      0.99        76
#            5       0.94      0.96      0.95       108
#            6       0.98      0.99      0.98        89
#            7       0.99      0.99      0.99        78
#            8       0.95      0.88      0.92        92
#            9       0.96      0.93      0.95        92

#     accuracy                           0.97       899
#    macro avg       0.97      0.97      0.97       899
# weighted avg       0.97      0.97      0.97       899


# # Tuning hyper-parameters for recall

# Best parameters set found on development set:

# {'C': 1, 'kernel': 'linear'}

# Grid scores on development set:

# 0.100 (+/-0.000) for {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
# 0.535 (+/-0.035) for {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.100 (+/-0.000) for {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
# 0.585 (+/-0.037) for {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.100 (+/-0.000) for {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
# 0.585 (+/-0.037) for {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.100 (+/-0.000) for {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}
# 0.585 (+/-0.037) for {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
# 0.971 (+/-0.010) for {'C': 1, 'kernel': 'linear'}
# 0.971 (+/-0.010) for {'C': 10, 'kernel': 'linear'}
# 0.971 (+/-0.010) for {'C': 100, 'kernel': 'linear'}
# 0.971 (+/-0.010) for {'C': 1000, 'kernel': 'linear'}

# Detailed classification report:

# The model is trained on the full development set.
# The scores are computed on the full evaluation set.

#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        89
#            1       0.96      0.96      0.96        90
#            2       0.98      0.99      0.98        92
#            3       0.96      0.99      0.97        93
#            4       0.99      1.00      0.99        76
#            5       0.94      0.96      0.95       108
#            6       0.98      0.99      0.98        89
#            7       0.99      0.99      0.99        78
#            8       0.95      0.88      0.92        92
#            9       0.96      0.93      0.95        92

#     accuracy                           0.97       899
#    macro avg       0.97      0.97      0.97       899
# weighted avg       0.97      0.97      0.97       899

# 附件3 python程序运行时间计时
# import time

# time_start=time.time()
# time_end=time.time()
# print('time cost',time_end-time_start,'s')

