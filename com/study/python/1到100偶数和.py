import turtle
turtle.done
i = 1
sum = 0
while i<=100:#满足条件才进入while循环
    if i % 2 == 0:
        sum = i+sum
    i+=1
print("从1加到100的偶数和为：",sum)
