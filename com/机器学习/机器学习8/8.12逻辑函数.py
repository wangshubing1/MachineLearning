# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:04:11 2016

@author: SumaiWong
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv

iris = pd.read_csv('iris.csv')
dummy = pd.get_dummies(iris['Species']) # 对Species生成哑变量
iris = pd.concat([iris, dummy], axis =1 )
iris = iris.iloc[0:100, :] # 截取前一百行样本

# 构建Logistic Regression , 对Species是否为setosa进行分类 setosa ~ Sepal.Length
# Y = g(BX) = 1/(1+exp(-BX))
def logit(x):
    return 1./(1+np.exp(-x))

temp = pd.DataFrame(iris.iloc[:, 0])
temp['x0'] = 1.
X = temp.iloc[:,[1,0]]
Y = iris['setosa'].reshape(len(iris), 1) #整理出X矩阵 和 Y矩阵

# 批量梯度下降法
m,n = X.shape #矩阵大小
alpha = 0.0065 #设定学习速率
theta_g = np.zeros((n,1)) #初始化参数
maxCycles = 3000 #迭代次数
J = pd.Series(np.arange(maxCycles, dtype = float)) #损失函数

for i in range(maxCycles):
    h = logit(dot(X, theta_g)) #估计值  
    J[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值      
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    theta_g -= alpha * grad
print(theta_g)
print(J.plot())

# 牛顿方法
theta_n = np.zeros((n,1)) #初始化参数
maxCycles = 10 #迭代次数
C = pd.Series(np.arange(maxCycles, dtype = float)) #损失函数
for i in range(maxCycles):
    h = logit(dot(X, theta_n)) #估计值 
    C[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值      
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    A =  h*(1-h)* np.eye(len(X)) 
    H = np.mat(X.T)* A * np.mat(X) #海瑟矩阵, H = X`AX
    theta_n -= inv(H)*grad
print(theta_n)
print(C.plot())