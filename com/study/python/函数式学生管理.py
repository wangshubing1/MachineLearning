import os

def read_stus():
    if os.path.exists(file_name):
        f =open(file_name,"r",encoding='utf-8')
        while True:
            student_str = f.readline()
            if student_str==" ":
                break
            else:
                student_info_list = student_str.split()
                student={"name":student_info_list[0],"age":student_info_list[1],"sex":student_info_list[2]}
                stus.append(student)
def write_stus_to_file():
    if os.path.exists(file_name):
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(file_name,"backup-"+file_name)
    f=open(file_name,"w",encoding='utf-8')
    for student in stus:
        student_str ="%s\t\t%s\t\t%s"%(student["name"],student["age"],student["sex"])
        f.write(student_str)
        f.write("\n")
    f.close()
def print_menu():
    print("="*50)
    print("学生名字管理系统".center(30))
    print("输入1：添加学生")
    print("输入2：查找学生")
    print("输入3：修改学生")
    print("输入4：删除学生")
    print("输入5：查看所有学生")
    print("输入6：退出")
def add_student():
    name = input("请输入要添加的学生姓名：")
    age = int(input("请输入要添加的学生年龄："))
    sex = input("请输入要添加的学生性别：")
    stu = {}
    stu["name"] = name
    stu["age"] = age
    stu["sex"] = sex
    stus.append(stu)
    print("添加成功")

def search_student(name):
    for item in stus:
        if item["name"] == name.strip():
            print("%s 学生存在"%name)
            print_student(item)
            #break
            return item
    else:
        print("查无%s此人！" % name)

def print_student(item):
    print("%s\t\t%s\t\t%s" % (item["name"], item["age"], item["sex"]))

def print_all_student():
    print("序号\t姓名\t年龄\t性别")
    for i, item in enumerate(stus, 1):
        print("%s\t" % i,end="")
        print_student(item)

def del_student(name):
    student =search_student(name)
    stus.remove(student)
def main():
    read_stus()
    print_menu()
    while True:
        operate=input("请输入你需要的操作代号：")
        if operate =="1":
            add_student()
            write_stus_to_file()
        if operate == "2":
            name = input("请输入要查找的学生姓名：")
            search_student(name)
        if operate == "3":
            print(stus)
            name = input("请输入要修改的学生姓名：")
            if name not in stus:
                print("查无%s此人！"%name)
            else:name=input("请输入修改后的名字：")
            print("修改后的名单为%s:"%stus)
        if operate == "4":
            name = input("请输入要删除的学生姓名：")
            del_student()
            print("删除成功！")
            write_stus_to_file()
        if operate == "5":
            print_all_student()
        if operate == "6":
            break
file_name="stus.txt"
backup_file="backup-stus.txt"
stus=[]
main()