#!/usr/bin/env python
# encoding: utf-8
'''
@author: 王树兵
@contact: wang_shubing@126.com
@file: json解析.py
@time: 2018/2/1 10:41
@desc:
'''
import json
def loadFont():
    f = open("test.json",encoding="utf-8")
    setting=json.load(f)
    return setting
t=loadFont()
print(t)
