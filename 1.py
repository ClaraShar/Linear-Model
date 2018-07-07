# -*- coding:utf-8 -*-
   #导入模块函数将数据集转化格式
from numpy import genfromtxt
import numpy as np   #导入模块numpy，中间需要用到矩阵的运算
from sklearn import datasets,linear_model  #导入数据集，并且导入线性回归模型函数

dataPath = r"/home/clara/study/linear regression model/train.txt"  #初始化数据集的文件路径
delivaryData = np.genfromtxt(dataPath,delimiter='\t')  #将数据集文件中的数据转换为矩阵形式

print("data")
print(delivaryData)

X = delivaryData[:,:-1] #提取数据集中所有行，并且除了倒数第一列以外的数据项
Y = delivaryData[:,-1]  #提取数据集中所有行，并且只有倒数第一列的数据项

print("特征矩阵")
print(X)
print("标记矩阵")
print(Y)

regr = linear_model.LinearRegression()  #创建线性回归模型对象

regr.fit(X,Y)  #将特征矩阵和标记矩阵传入模型器中训练模型

print("x的系数：",regr.coef_)   #输出模型的有关x的参数，b1,b2...
print("截距：",regr.intercept_)  #输出线性模型中的截距，b0

predict_x = [[0,0,0],]  #给出测试矩阵
predict_y = regr.predict(predict_x)  #预测
print("预测结果：",predict_y)
'''
testPath = r"/home/xiayulu/linear regression model/test.txt"
predict_x = np.genfromtxt(testPath,delimiter='\t') #给出测试矩阵
x1 = predict_x[:,:0]
predict_y = regr.predict(x1) #预测
print("预测结果：",predict_y)
'''
