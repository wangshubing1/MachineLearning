age = int(input("输入年龄："))
sex=input("输入性别:")
#and表示并且。or表示或者。not表示不满足后面的条件
if age>=18 and sex =="男":
    print("去做苦力吧！")
elif age<18 or sex =="女":
    print("不用干活啊！")
elif not(sex=='男'or sex=='女'):
    print("你丫泰国来的！")
else:
    pass#以后要填充代码
age1=int(input("请输入值:"))
if age1:
    print("哈哈")
else:
    print("呵呵")