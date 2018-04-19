def fab(n):
    if n<=1:
        return 1
    else:return fab(n-1)+fab(n-2)
nums=[]
for i in range(20):
    nums.append(fab(i))
print(fab(10))
print(nums)