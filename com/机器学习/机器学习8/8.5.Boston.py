#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
import sklearn.datasets
from pprint import pprint
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
#import exceptions


def not_empty(s):
    return s != ''


if __name__ == "__main__":
    # 忽略警告信息
    warnings.filterwarnings(action='ignore')
    # 设置打印选项 --suppress消除小的数字使用科学记数法
    np.set_printoptions(suppress=True)
    # header=None时，即指明原始文件数据没有列索引，这样read_csv为自动加上列索引，除非你给定列索引的名字
    file_data = pd.read_csv('housing.data', header=None)
    # a = np.array([float(s) for s in str if s != ''])
    # 根据读入的文件内容创建二维数组,第一个参数为行数,第二个参数为列数
    data = np.empty((len(file_data), 14))
    for i, d in enumerate(file_data.values):
    # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回
    # filter()函数接收一个函数 f 和一个list，这个函数 f 的作用是对每个元素进行判断，返回 True或 False，filter()根据判断结果自动过滤掉不符合条件的元素，返回由符合条件元素组成的新list
        #print(d)
        # 首先过滤掉非空元素,然后将列表中的每个元素转换成float类型
        d = list(map(float, filter(not_empty, d[0].split(' '))))
        print(d)
        data[i] = d
    # array是按照从左至右的顺序切分x是特征向量 1-13列  y是标签列 14列
    x, y = np.split(data, (13, ), axis=1)
    data = sklearn.datasets.load_boston()
    x = np.array(data.data)
    y = np.array(data.target)
    print ('样本个数：%d, 特征个数：%d' % x.shape)
    print (y.shape)
    y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    # Pipeline可以将许多算法模型串联起来，比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流。主要带来两点好处：
    # 1. 直接调用fit和predict方法来对pipeline中的所有算法模型进行训练和预测。
    # 2. 可以结合grid search对参数进行选择开始建模...
    model = Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                fit_intercept=False, max_iter=1e3, cv=3))
    ])
    print('开始建模...')
    model.fit(x_train, y_train)
    linear = model.get_params('linear')['linear']
    print('超参数：', linear.alpha_)
    print ('L1 ratio：', linear.l1_ratio_)
    # print u'系数：', linear.coef_.ravel()

    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('均方误差：', mse)

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label=u'真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label=u'估计值')
    plt.legend(loc='best')
    plt.title(u'波士顿房价预测', fontsize=18)
    plt.xlabel(u'样本编号', fontsize=15)
    plt.ylabel(u'房屋价格', fontsize=15)
    plt.grid()
    plt.show()
