#!/usr/bin/python
# encoding: utf-8
from pyspark import SparkContext
import requests
from bs4 import BeautifulSoup
"""
@author : 王树兵
@contact : wang_shubing@126.com
@File : wordsCount.py
@time : 2017/11/20 15:58
"""

#设置列表页URL的固定部分
url='http://sh.lianjia.com/ershoufang/d'

#设置请求头部信息
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
    'Accept':'text/html;q=0.9,*/*;q=0.8',
    'Accept-Charset':'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding':'gzip',
    'Connection':'close',
    'Referer':'http://www.baidu.com/'}

listurl=[]
for num in range(1,101):
    newurl=url+str(num)
    listurl.append(newurl)

sc = SparkContext(appName="sparkpachong")
urlrdd = sc.parallelize(listurl)

tp=[]
fjhtmlrdd=urlrdd.map(lambda x:BeautifulSoup(requests.get(url=x,headers=headers).text,'html.parser').find_all('span',attrs={'class':'total-price strong-num'}))
# ,lambda x:
# BeautifulSoup(
# requests.get(url=x, headers=headers).text, 'html.parser').
# find_all('span', attrs={'class': 'total-price strong-num'})


output = fjhtmlrdd.collect()
for (w) in output:
    for a in w:
        tp.append(a.string)
        print(a.string )
print(tp)
sc.stop()
