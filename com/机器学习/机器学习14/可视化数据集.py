#!/usr/bin/env python
# encoding: utf-8
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: 可视化数据集.py
@time: 2018/1/31 17:15
@desc:
'''
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet( fileName ):
    """
    读取数据
    Parameters:
    fileName - 文件名
    Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
    """
    dataMat = []
    labelMat = []
    fr =open(fileName,encoding="utf-8")
    for line in fr.readlines():                                      #逐行读取，滤除空格等
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1])])#添加数据
        labelMat.append(float(lineArr[2]))                           #添加标签
    return dataMat,labelMat
def showDataSet(dataMat,labelMat):
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])  # 负样本散点图
    plt.show()


if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSetRBF.txt')  # 加载训练集
    showDataSet(dataArr,labelArr)