def test1(x,y,*args):
    print(x,y)
    print(args)
    sum =x+y
    for i in args:
        sum+=i
    print(sum)
test1(2,3,4,5,6,7)


def test2(x,y):
    x=x.replace("c","C")
    y.append(10)
    print("x的变量指向的内存地址为：%s"%id(x))
    print("y的变量指向的内存地址为：%s" % id(y))
a="abcdefg"
b=[1,2,3]
print("a的变量指向的内存地址为：%s"%id(a))
print("b的变量指向的内存地址为：%s" % id(b))
test2(a,b)