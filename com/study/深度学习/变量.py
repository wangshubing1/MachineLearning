#1.
import tensorflow as tf
x=tf.Variable([1,2])
a=tf.constant([3,3])
#增加一个减法op
sub=tf.subtract(x,a)
#增加一个加法op
add=tf.add(x,sub)
#初始化所有变量
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)#变量初始化操作
    print(sess.run(sub))
    print(sess.run(add))
print("==="*30)
#2.
#创建一个变量初始化为零
state=tf.Variable(0,name='counter')
#创建一个op，作用是使state加1
new_value=tf.add(state,1)
#赋值op
update=tf.assign(state,new_value)
#变量初始化
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))