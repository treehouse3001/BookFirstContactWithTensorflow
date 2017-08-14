# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 02:21:55 2017

@author: Maxi
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1,y1])
    
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data, y_data)
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))


optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(10):
    sess.run(train)

    print(step, sess.run(W), sess.run(b))
    print(step, sess.run(loss))

    plt.scatter(x_data, y_data)
    plt.plot(x_data,sess.run(W) * x_data + sess.run(b), color = 'black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2,2)
    plt.ylim(0.1,0.6)
    plt.show()