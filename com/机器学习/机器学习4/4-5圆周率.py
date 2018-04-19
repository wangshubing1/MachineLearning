# /usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: Adaboost实战.py
@time: 2018/1/25 17:24
@desc:
'''
import numpy as np

if __name__ == '__main__':
    print (np.sqrt(6 * np.sum(1 / np.arange(1, 100000, dtype=np.float) ** 2)))
