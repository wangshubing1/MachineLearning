class A(object):
    name="zs"
    def test1(self):
        print("--------------A 的test1方法")
    @classmethod#类方法一定要在方法的上面加上一个修饰器（java注解）
    def test2(cls):
        cls.name="ww"
        print("---------------A 的test2方法")
    @staticmethod
    def test3():
        A.name="ls"
        print("--------------A的test3静态方法")
a=A()
a.test2()
A.test2()
A.test3()
print(A.name)