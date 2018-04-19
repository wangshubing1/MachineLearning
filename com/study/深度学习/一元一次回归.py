import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
#从-0.5到0.5之间生成20个随机点
x_data = np.linspace(-0.5, 0.5, 20)[:, np.newaxis]
y_data = x_data +1
plt.figure()
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(x_data,y_data,'r-',x_data,y_data,'go',lw=1)
plt.title(u'一元一次回归函数', fontsize=18)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y(x)', fontsize=15)
plt.grid(True,ls=":")
plt.show()
