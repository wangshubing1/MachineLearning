#定义了一个类
class Car:
    def start(self):
        print("汽车启动")
    def print_car_info(self):
        print("车的名字是：%s，颜色为：%s"%(self.name,self.color))
c=Car()#构建一个对象
c.name="奔驰"#给c这个对象一个属性
c.color="金色"
c.print_car_info()
c.start()