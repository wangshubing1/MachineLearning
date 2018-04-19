import pandas as pd
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
# %config InlineBackend.figure_formats = {'pdf',}


sns.set_context('notebook')
sns.set_style('white')
def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[1:6,:])
    return(data)


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)




#定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
# 定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]

# 求解梯度
def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * X.T.dot(h - y)
    return (grad.flatten())



data = loadtxt('data1.txt', delimiter=',')
X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)
res = minimize(costFunction, initial_theta, args=(X,y), jac=gradient, options={'maxiter':400})
print("res:",res)
#做一下预测吧
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
#咱们来看看考试1得分45，考试2得分85的同学通过概率有多高
print("-------------------------------------------------")
test=sigmoid(np.array([1, 45, 85]).dot(res.x.T))
print(test)
#画决策边界
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plt.show()
#****************************************************************************
#加正则化项的逻辑斯特回归
data2 = loaddata('data2.txt', ',')
# 拿到X和y
y1 = np.c_[data2[:,2]]
X1 = data2[:,0:2]
# 画个图
plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
plt.show()
#咱们整一点多项式特征出来(最高6阶)
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:,0:2])
# 看看形状(特征映射后x有多少维了)
print(XX.shape)


# 定义损失函数
def costFunctionReg(theta, reg, *args):
    m = y1.size
    h = sigmoid(XX.dot(theta))

    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y1) + np.log(1 - h).T.dot(1 - y1)) + (reg / (2.0 * m)) * np.sum(
        np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def gradientReg(theta, reg, *args):
    m = y1.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))

    grad = (1.0 / m) * XX.T.dot(h - y1) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())
initial_theta = np.zeros(XX.shape[1])
costFunctionReg(initial_theta, 1, XX, y1)
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

# 决策边界，咱们分别来看看正则化系数lambda太大太小分别会出现什么情况
# Lambda = 0 : 就是没有正则化，这样的话，就过拟合咯
# Lambda = 1 : 这才是正确的打开方式
# Lambda = 100 : 卧槽，正则化项太激进，导致基本就没拟合出决策边界

for i, C in enumerate([0.0, 1.0, 100.0]):
    # 最优化 costFunctionReg
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y1), jac=gradientReg, options={'maxiter': 3000})

    # 准确率
    accuracy = 100.0 * sum(predict(res2.x, XX) == y1.ravel()) / y1.size

    # 对X,y的散列绘图
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])

    # 画出决策边界
    x1_min, x1_max = X1[:, 0].min(), X1[:, 0].max(),
    x2_min, x2_max = X1[:, 1].min(), X1[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))
show()