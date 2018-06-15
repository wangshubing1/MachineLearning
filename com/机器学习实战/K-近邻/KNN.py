from numpy import *
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
def createDataSet():
    group =array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group,labels
'''
分类:inX
训练样本集:dataSet
标签向量:labels
选择最近邻居的数目:k
'''
#K-近邻算法
def classify0 (inX,dataSet,labels,k):
    # 获取训练数据集的行数
    dataSetSize = dataSet.shape[0]
    #---------------欧氏距离计算-----------------
    # p19
    #各个函数均是以矩阵形式保存
    #tile():inX沿各个维度的复制次数
    diffMat =tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat =diffMat**2
    # .sum()运行加函数，参数axis=1表示矩阵每一行的各个值相加和
    sqDistances =sqDiffMat.sum(axis=1)
    distances =sqDistances**0.5
    #--------------------------------------------
    #获取排序（有小到大）后的距离值的索引（序号）
    sortedDistIndicies =distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel =labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount =sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr =open(filename)
    arrayOlines =fr.readlines()
    numberOfLines =len(arrayOlines)
    returnMat =zeros((numberOfLines,3))
    classLabelVector =[]
    index=0
    for line in arrayOlines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat ,classLabelVector
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals =dataSet.max(0)
    ranges=maxVals - minVals
    normDataSet=zeros(shape(dataSet))
    m =dataSet.shape[0]
    normDataSet= dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
'''
分类器针对约会网站的测试
'''
def datingClasssTest():
    hoRatio =0.10
    datatingDataMat,datatingLabels =file2matrix('datingTestSet2.txt')
    normMat ,ranges,minVals=autoNorm(datatingDataMat)
    m = normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datatingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is:%d"\
              %(classifierResult,datatingLabels[i]))
        if(classifierResult !=datatingLabels[i]):
            errorCount +=1.0
            print("the total error rate is; %f"%(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList =["不喜欢","魅力一般","极具魅力"]
    percentTats=float(input("玩电子游戏的时间百分比？"))
    ffMiles =float(input("每年飞行里程数？"))
    iceCream =float(input("冰淇淋每年消耗一升？"))
    datatingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,range,minVals=autoNorm(datatingDataMat)
    inArr =array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person: ",resultList[classifierResult - 1])
if __name__=='__main__':
    group =array([[1.,1.1],[1.,1.0],[0.,0.],[0.,0.1]])
    labels =['A','A','B','B']
    print(classify0([2,5],group,labels,3))
    print(createDataSet())

    datatingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datatingDataMat[:, 1], datatingDataMat[:, 2])
    # 加入颜色区分
    ax.scatter(datatingDataMat[:, 1], datatingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # ax.scatter(datatingDataMat[:, 0], datatingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # 设置中文类型
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.ylabel('每周消费的冰淇淋公升数',fontsize=15)
    plt.xlabel('玩视频游戏所耗时间百分比',fontsize=15)
    #plt.ylabel('玩视频游戏所耗时间百分比', fontsize=15)
    #plt.xlabel('每年获取的飞行常客里程数', fontsize=15)
    plt.show()

    normMat, ranges, minVals=autoNorm(datatingDataMat)
    print('normMat:\n',normMat,'\n','ranges:\n',ranges,'\n','minVals:\n',minVals)

    print(datingClasssTest())
    print(classifyPerson())

