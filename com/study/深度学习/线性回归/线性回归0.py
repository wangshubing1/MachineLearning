import numpy as np
import matplotlib.pyplot as plt

x = np.array([30,35,37,	59,	70,	76,	88,	100]).astype(np.float32)
print("x=:",x)
y = np.array([1100,	1423,1377,1800,2304,2588,3495,4839]).astype(np.float32)
print("====="*30)
print("y=:",y)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)
#实际算法
for i in range(0,len(x)):
    x[i] = (x[i] - x_min)/(x_max - x_min)
    y[i] = (y[i] - y_min)/(y_max - y_min)
print("****"*30)
print("new(x):%s \nnew(y):%s"%(x,y))
a = 1
b = -0.15
#X轴与Y轴
y_ , x_ = np.array([0,1]),np.array([0,1])
#回归方程式
yp = a*x +b
r = sum(np.square(np.round(yp-y,4)))#square取平方
print("****"*30)
#就是新的x减掉新的y保留小数点后4位
print(np.round(yp-y,4))
print("****"*30)
print(np.square(np.round(yp-y,4)))
print(r/16)
plt.scatter(x,y)
plt.xlabel(u"X")
plt.ylabel(u"Y")
plt.plot(x,yp,"r-",color='green')
plt.show()
#程序运行5秒
#plt.pause(5)