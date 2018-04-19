import numpy as np
import random
print("==="*30)
print(np.__version__)
arr = np.random.random([6,3])
print("==="*30)
print(arr)
arr1=np.arange(16).reshape(2,2,4)
print("==="*30)
print(arr1)
arr2=arr1.transpose((1,0,2))
print("==="*30)
print(arr2)
a = [1, 2, 3, 4]
b = np.array(a)
print("==="*30)
print(b)# array([1, 2, 3, 4])
print("==="*30)
print(type(b))# <type 'numpy.ndarray'>
print("==="*30)
print(b.shape)# (4,)
print(b.argmax())  #找出最大值的位置             	# 3
print(b.max())                   	# 4
print(b.mean())                    # 2.5
print("==="*30)

a1 = np.array([3, 1, 2, 4, 6, 1])
print(np.argmax(a1, axis=0))
print("==="*30)
a2 = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])
print(np.argmax(a2, axis=0))
print("*"*30)
print(np.argmax(a2, axis=1))
print("==="*30)
a3 = np.array([
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 1]
              ],

              [
                  [-1, 5, -5, 2],
                  [9, 6, 2, 8],
                  [3, 7, 9, 1]
              ]
            ])
print(np.max(a3, axis=0))
print("---"*30)
print(np.argmax(a3, axis=0))
print("*"*30)
print(np.max(a3, axis=1))
print("---"*30)
print(np.argmax(a3, axis=1))
print("*"*30)
print(np.max(a3, axis=2))
print("---"*30)
print(np.argmax(a3, axis=2))


print("==="*30)
c = [[1, 2, 8], [3, 4, 6]]  	# 二维列表
d = np.array(c)         	# 二维numpy数组
d.shape                   	# (2, 2)
d.size                   	# 4
print(d.max(axis=0))            	# 找维度0，也就是最后一个维度上的最大值，array([3, 4])
print(d.max(axis=1))            	# 找维度1，也就是倒数第二个维度上的最大值，array([2, 4])
print(d.mean(axis=0))          	# 找维度0，也就是第一个维度上的均值，array([ 2.,  3.])
print(d.flatten())              	# 展开一个numpy数组为1维数组，array([1, 2, 3, 4])
print(np.ravel(c))               # 展开一个可以解析的结构为1维数组，array([1, 2, 3, 4])
print("==="*30)
# 3x3的浮点型2维数组，并且初始化所有元素值为1
e = np.ones((3, 3), dtype=np.float)
print(e)
print("==="*30)
# 创建一个一维数组，元素值是把3重复4次，array([3, 3, 3, 3])
f = np.repeat(3, 4)
print(f)
print("==="*30)
# 2x2x3的无符号8位整型3维数组，并且初始化所有元素值为0
g = np.zeros((2, 2, 3), dtype=np.uint8)
print(g)
print(g.shape)  # (2, 2, 3)
print("==="*30)
h = g.astype(np.float)  # 用另一种类型表示
print(h)
print("==="*30)
l = np.arange(10)  # 类似range，array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(1)
print("==="*30)
m = np.linspace(0, 6, 5)  # 等差数列，0到6之间5个取值，array([ 0., 1.5, 3., 4.5, 6.])
print(m)
print("==="*30)
p = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]]
)

np.save('p.npy', p)  # 保存到文件
q = np.load('p.npy')  # 从文件读取
print(q)