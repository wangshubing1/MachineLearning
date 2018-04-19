import urllib.request

url="http://douban.com/"
requset=urllib.request.Request(url)
response =urllib.request.urlopen(requset)
data=response.read()
data =data.decode("utf-8")
print(data)
print(type(response))
print(response.geturl())
print(response.info())
print(response.getcode())