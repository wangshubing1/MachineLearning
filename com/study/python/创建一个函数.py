#定义一个函数
def print_hello():
    print("hello,python!")
#调用函数
print_hello()
#定义一个计算圆的面积
def test1(r):
    s = 3.14*(r**2)
    print("圆的面积为：%s"%s)
test1(10)
#计算三个数的平均值
def avg_3(a,b,c):
    if is_number(a) and is_number(a) and is_number(a):
        return (a+b+c)/3
    else:print("你传的实参有误")
def is_number(a):
    if not isinstance(a,(int,float)):
        return False
    else:return True
avg=avg_3(25,24,32)
print(avg)