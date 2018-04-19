class Person:
    def __init__(self,name,age):
        self.name =name
        self.age=age
    def introduce(self):
        print("姓名：%s,年龄为：%s"%(self.name,self.age))
    def __str__(self):
        return "姓名：%s,年龄为：%s"%(self.name,self.age)
p1 =Person("zhangsan",19)
p1.introduce()
print(p1)