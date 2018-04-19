import tensorflow as tf
#创建一个常量op
m1 = tf.constant([[3,3]])
#创建一个常量op
m2 = tf.constant([[2],[3]])
#创建一个矩阵乘法op，把m1和m2传入
product = tf.matmul(m1,m2)
#这是因为图必须在会话中执行
print(product)
#方法一
'''#定义一个会话，启动默认图
sess=tf.Session()
#调用sess的run方法来执行矩阵乘法op
#run（product）触发了图中3个op
result=sess.run(product)
print(result)
sess.close()
'''
#方法二
with tf.Session() as sess:
    #调用sess的run方法来执行矩阵乘法op
    #run（product）触发了图中3个op
    result=sess.run(product)
    print(result)