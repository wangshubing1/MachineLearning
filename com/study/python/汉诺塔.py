B=[] #设置操作过程列表
def move(n, a, b, c):
    if n==1:
        buzhou=a+str(n)+'-->'+c+str(n) #一个圆盘需要从A到C操作步骤
        B.append(buzhou) #向列表中添加操作步骤
        return
    move(n-1,a,c,b) #将A柱的n-1个盘移到B柱
    buzhou=a+str(n)+'-->'+c+str(n) #将A柱的第n个盘移到C柱操作步骤
    B.append(buzhou) #向列表中添加操作步骤
    move(1,a,b,c) #将A柱上最后一个盘移到C柱
    move(n-1,b,a,c) #将过渡柱子B上n-1个圆盘B移动到目标柱子C
move(3,'A','B','C') #2**64-1，64次太大，这里用6个盘子
print('共需操作'+str(len(B))+'次','操作过程为',B)#计算6个盘子的步骤数及操作过程

def hanoi(n,x,y,z):
    if n==1:
        print(x,'-->',z)
    else:
        hanoi(n-1,x,z,y)#将前n-1个盘子从x移动到y上
        hanoi(1,x,y,z)#将最底下的最后一个盘子从x移动到z上
        hanoi(n-1,y,x,z)#将y上的n-1个盘子移动到z上
n=int(input('请输入汉诺塔的层数：'))
hanoi(n,'x','y','z')