#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV#GridSearchCV模块，能够在指定的范围内自动搜索具有不同超参数的不同


if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('Advertising.csv')# TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    #print (x)
    #print (y)
    #分割测试数据和训练数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.8)
    #Lasso回归
    #model = Lasso()
    #岭回归
    model = Ridge()
    alpha_can = np.logspace(-3, 2, 10)#alpha参数集 10^-3~10^2的等比数列的10个数
    #不让alpha_can输出为科学计数法（算不上核心代码）
    np.set_printoptions(suppress=True)
    print ('alpha_can = ', alpha_can)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)#其中cv可以是整数或者交叉验证生成器或一个可迭代器，cv参数对应4种输入：
    #1、None:默认参数，哈数会使用默认的3折交叉验证
    #2、整数k：K折交叉验证。对于分类任务，使用stratifiedFold(类别平衡，每类的训练集占比一样多），对于其他任务，使用Kfold
    #3、交叉验证生成器：得自己写生成器，头疼，略
    #4、可以生成训练集与测试集的迭代器：同上，略
    lasso_model.fit(x_train, y_train)
    print ('训练得到最优参数：\n', lasso_model.best_params_)
    print("+++++++++++++++")
    #画图时将测试用的Y值从小到大排列
    order = y_test.argsort(axis=0)
    #实际值
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    #模型计算出来的值
    y_hat = lasso_model.predict(x_test)
    print (lasso_model.score(x_test, y_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
