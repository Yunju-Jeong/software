import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('고속도로교통사고현황.csv', encoding='CP949')
list1 = df.values.tolist()
ar = np.array(list1)
xData = ar[:, 0]-2000
yData = ar[:, 1]

W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b
cost = tf.reduce_mean(tf.square(H-Y))
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5001):
    sess.run(train, feed_dict={X : xData, Y : yData})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={X : xData, Y : yData}), sess.run(W), sess.run(b))

print(sess.run(H, feed_dict={X : [16]}))

hY = sess.run([(W * x + b) for x in xData])
plt.plot(xData, yData)
plt.plot(xData, hY, 'red')
plt.show()