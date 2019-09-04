import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)


# 'module' object has no attribute 'mul'
output = tf.mul(input1, input2)

with tf.compat.v1.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))



