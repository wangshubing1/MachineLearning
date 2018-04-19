names=["zs","ls","ww","tl"]
names2=("zs","ls","ww","tl")
j=0
print("序号\t姓名")
for i in names:
    j+=1
    print("%d\t\t%s"%(j,i))
print("="*30)
#枚举！
for i,item in enumerate(names,1):
    print("%d\t\t%s" % (i, item))