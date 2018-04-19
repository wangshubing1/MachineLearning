#!/usr/bin/env python
# encoding: utf-8
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: AllElectronics.py
@time: 2018/1/23 11:42
@desc:
'''

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
import pydotplus


allElectronicsData = open('AllElectronics.csv')
reader = csv.reader(allElectronicsData)
headers = next(reader) #取出标题

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])  #取最后一列
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]  #将数据保存为字典，不含最后一列
    featureList.append(rowDict)


# 非数值类型变为数值矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# yes和no转化为0和1
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# 使用决策树进行分类clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# 生成决策树pdf图
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris2.pdf")

#验证模型
#拿第一行数据进行修改，测试模型结果
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1, -1)) #需要将数据reshape(1, -1)处理
print("predictedY: " + str(predictedY))