#!/usr/bin/env python
# encoding: utf-8
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: Iris_Adaboost.py
@time: 2018/1/25 13:45
@desc:
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x_prime = data[list(range(4))]
    y = pd.Categorical(data[4]).codes
    x_prime,x_test,y,y_test=train_test_split(x_prime,y,train_size=0.7,random_state=0)

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for i, pair in enumerate(feature_pairs):
        # 准备数据
        x = x_prime[pair]

        # 随机森林
        base_estimator=DecisionTreeClassifier(criterion="gini",max_depth=4,min_samples_split=4)
        clf = AdaBoostClassifier(base_estimator=base_estimator,n_estimators=20,learning_rate=0.5)
        clf.fit(x, y.ravel())

        # 画图
        N, M = 50, 50  # 横纵各采样多少个值
        x1_min, x2_min = x.min()
        x1_max, x2_max = x.max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
        x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

        # 训练集上的预测结果
        print( '特征：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]],)
        print( '\t训练集准确率：%.2f%%'%(100*accuracy_score(y,clf.predict(x))))
        print('\t测试集准确率：%.2f%%' % (100 * accuracy_score(y_test, clf.predict(x_test[pair]))))


        # 显示
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_show = clf.predict(x_show)  # 预测值
        y_show= y_show.reshape(x1.shape)  # 使之与输入的形状相同
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_show, cmap=cm_light)  # 预测值
        plt.scatter(x[pair[0]], x[pair[1]], c=y.ravel(), edgecolors='k', cmap=cm_dark)  # 样本
        plt.scatter(x_test[pair[0]], x_test[pair[1]],c=y_test.ravel(),marker="*",s=60,edgecolors='k',zorder=10,cmap=cm_dark)
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
    plt.tight_layout(2.5)
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Adaboost对鸢尾花数据的两特征组合的分类结果', fontsize=18)
    plt.show()