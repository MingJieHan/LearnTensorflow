import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 4


### create tensorflow structure start ###
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
### create tensorflow structure end ###


#sess = tf.Session()
sess = tf.compat.v1.Session()
sess.run(init)  #Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights), sess.run(biases))

