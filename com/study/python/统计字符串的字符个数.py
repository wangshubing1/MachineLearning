str = input("请输入一个字符串:")
res ={}
for i in str:
    if i in res:
        res[i]=res[i]+1
    else:
        res[i]=1
print(res)