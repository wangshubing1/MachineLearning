import os
#从。。。。目录下找到包含hello的py文件是哪些
file_list=[]
#递归函数，该函数中所有的文件路径全部采用绝对路径，parent_dir：文件所在父目录的绝对路径，file_name表示当前你要处理的文件或者目录
def find_hello(parent_dir,file_name):
    file_abspath = os.path.join(parent_dir,file_name)#当前正在处理的文件或者目录的绝对路径
    if os.path.isdir(file_abspath):#如果传入放入文件是一个目录
        for f in os.listdir(file_abspath):#进入目录，列出该目录下所有文件列表
            find_hello(file_abspath,f)#递归调用自己本身的函数
    else:
        if file_abspath.endswith(".py"):
            if read_and_find_hello(file_abspath):#读取该py结尾的文件，并且看看文件内容是否包含hello
                file_list.append(file_abspath)
#主要功能：读取py结尾的文件，并且看看文件内容中是否包含hellp，如果有返回True 否则返回false
def read_and_find_hello(py_file):
    flag =False#定义一个标记变量
    f=open(py_file,encoding='utf-8')#打开文件
    while True:
        line=f.readline()#读取其中一行
        if line==" ":
            break
        elif "hello" in line:
            flag =True
            break
        if not line:
            break
    f.close()
    return flag
find_hello("D:/PycharmProjects/test20171218/com/study/","python")
print(file_list)