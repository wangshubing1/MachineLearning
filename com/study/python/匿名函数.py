#匿名函数
lambda x,y:x+y
#匿名函数调用
func=lambda a,b:a+b
print(func(22,66))
def tset(a,b):
    result =func(a,b)
    return result
print(tset(15,6))
stus=[{"name":"zs","age":"22"},{"name":"lao","age":"26"},{"name":"tl","age":"22"}]
a1=[222,2,34,4,5,56,56,76,7,624,22]
#a1.sort(reverse=True)
#a1.reverse()
#print(a1)
stus.sort(key=lambda x:x["name"])#匿名函数功能：返回列表中每个元素的（字典）的“name”对应的值
print(stus)
def test2(a,b,func):
    result = func(a, b)
    return result
func_new =input("请输入你的操作:")
#动态语言
func_new =eval(func_new)#把字符串转换成表达式
print(test2(22,33,func_new))