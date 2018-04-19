#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint
from sklearn import metrics

if __name__ == "__main__":
    path = 'Advertising.csv'
    # # 手写读取数据
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # pprint(x)
    # pprint(y)
    # x = np.array(x)
    # y = np.array(y)

    # Python自带库
    # f = file(path, 'r')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p
    # print '\n\n===============\n\n'

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    #x = data[['TV', 'Radio']]
    y = data['Sales']
    #我们看看数据的维度(结果有200个样本，每个样本有5列)
    print(data.shape)
    print("************"*5)
    print ("x=:\n",x)
    print("========================")
    print ("y=:\n",y)

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制1
    plt.figure(facecolor='w')
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    plt.xlabel(u'广告花费', fontsize=16)
    plt.ylabel(u'销售额', fontsize=16)
    plt.title(u'广告花费与销售额对比数据', fontsize=20)
    plt.grid()
    plt.show()

    # 绘制2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()
    #划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)#表示0.8作为训练数据，0.2作为测试数据
    print (type(x_test))
    #查看下训练集和测试集的维度：
    print (x_train.shape, y_train.shape)
    print("========================"*3)
    linreg = LinearRegression()#线性回归
    model = linreg.fit(x_train, y_train)#用训练集来拟合出线性回归模型
    print (model)
    #我们看看我们的需要的模型系数结果
    print (linreg.coef_, linreg.intercept_)#系数,截距
    #argsort函数返回的是数组值从小到大的索引值，沿第一轴排序（向下）
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    #模型拟合测试集
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error（均方误差）
    rmse = np.sqrt(mse)  # Root Mean Squared Error（均方根误差）
    #用scikit-learn计算 MSE,RMSE
    smse=metrics.mean_squared_error(y_test,y_hat)
    srmse=np.sqrt(smse)
    print("+++++++++++++++++++++均方误差，均方根误差+++++++++++++++++++++++++++++++")
    print('MSE = ', mse,"sMSE=",smse)
    print('RMSE = ', rmse,"sRMSE=",srmse)
    print ('R2 = ', linreg.score(x_train, y_train))
    print ('R2 = ', linreg.score(x_test, y_test))
    #画图
    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True)
    plt.show()
