import tensorflow as tf
#Fetch  可以同时运行多个op
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

add=tf.add(input2,input3)
mul=tf.multiply(input1,add)

with tf.Session() as sess:
    result=sess.run([mul,add])#运行两个op
    print(result)