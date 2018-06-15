#--*-- coding:utf-8 --*--
from selenium import webdriver
import time
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import re
import urllib.request
import warnings

warnings.filterwarnings("ignore")
#基于PhantomJS创建一个浏览器，并且设置一下用户代理，否则可能出现界面不兼容的情况
dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"]=("Mozilla/5.0(Windows NT 6.1;Win64;x64)")
browser = webdriver.PhantomJS(desired_capabilities = dcap)

#通过PhantomJS打开相关动漫网页，将相关动漫图片地址触发出来
#打开动漫的第一页
browser.get('http://ac.qq.com/ComicView/index/id/539443/cid/1')
for i in range(10):
    js='window.scrollTo('+str(i*1280)+','+str((i+1)*1280)+')'  #把内容滚动到指定的坐标
    browser.execute_script(js)
    time.sleep(1)
#将打开的界面截图保存，方便观察
a=browser.get_screenshot_as_file(r'C:\Users\king/Desktop\test.jpg')
#获取当前页面所有源码(此时包含出发出来的异步加载的资源)
data=browser.page_source
#将相关网页源码写入本地文件中，方便分析
fh=open(r'C:\Users\king/Desktop\dongman.html','w',encoding='utf-8')
fh.write(data)
fh.close()
browser.quit()
#构造正则表达式提取动漫图片资源网址
pat='<img src="(http://ac.tc.qq.com.store_file_download.buid=.*?name=.*?).jpg"'
#获取所有动漫图片资源网站
allid=re.compile(pat).findall(data)
for i in range(0,len(allid)):
    #得到当前网址
    thisurl=allid[i]
    #去除网址中的多余元素amp
    #thisurl2=thisurl.replace("amp;","")+".jpg"
    #输出当前爬取的网址
    print(thisurl)
    #设置将动漫存储到本地目录
    localpath='C:/Users/king/Desktop/dongman/'+str(i)+'.jpg'
    #通过urllib对动漫图片资源进行爬取
    urllib.request.urlretrieve(thisurl,filename=localpath)