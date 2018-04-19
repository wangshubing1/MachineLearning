#!/usr/bin/env python
# encoding: utf-8
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: 基尼系数.py
@time: 2018/1/22 9:39
@desc:
'''
import numpy as np
import matplotlib.pyplot as plt
if __name__ =="__main__":
    p=np.arange(0.001,1,0.001,dtype=np.float)
    gini =2*p*(1-p)
    h=-(p*np.log2(p)+(1-p)*np.log2(1-p))/2
    err=1-np.max(np.vstack((p,1-p)),0)
    plt.plot(p,h,"b-",linewidth=2,label="Entropy")
    plt.plot(p,gini,"r-",linewidth=2,label="Gini")
    plt.plot(p, err, "g-", linewidth=2, label="Error")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.show()
