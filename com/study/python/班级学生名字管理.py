print("="*50)
print("学生名字管理系统".center(30))
print("输入1：添加学生")
print("输入2：查找学生")
print("输入3：修改学生")
print("输入4：删除学生")
print("输入5：退出")

stus=[]
while True:
    operate=input("请输入你需要的操作代号：")
    if operate =="1":
        name =input("请输入要添加的学生姓名：")
        stus.append(name.strip())
        print(stus)
    if operate == "2":
        name = input("请输入要查找的学生姓名：")
        if name not in stus:
            print("查无%s此人！"%name)
        else:print(name)
    if operate == "3":
        print(stus)
        name = input("请输入要修改的学生姓名：")
        if name not in stus:
            print("查无%s此人！"%name)
        else:name=input("请输入修改后的名字：")
        print("修改后的名单为%s:"%stus)
    if operate == "4":
        name = input("请输入要删除的学生姓名：")
        if name not in stus:
            print("查无%s此人！"%name)
            continue
        else:stus.remove(name)
        print("删除成功")
    if operate == "5":
        break
