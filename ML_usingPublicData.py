import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('고속도로교통사고현황.csv', encoding='CP949')
ar = df.to_numpy()
xData = ar[:, 0]-2000 # 연도에서 2000을 뺀다. (예) 2010 ==> 10
yData = ar[:, 1]      # 사고 수

W = tf.Variable(tf.random_uniform([1], -100, 100)) # W 변수 선언하고 랜덤 넘버로 초기화
b = tf.Variable(tf.random_uniform([1], -100, 100)) # b 변수 선언하고 랜덤 넘버로 초기화
X = tf.placeholder(tf.float32)                     # X 변수 선언
Y = tf.placeholder(tf.float32)                     # Y 변수 선언
H = W * X + b                                      # 선형 회귀 방정식 선언
cost = tf.reduce_mean(tf.square(H-Y))              # 비용함수식 선언
a = tf.Variable(0.01)                              # Learning rate 선언 및 초기화
optimizer = tf.train.GradientDescentOptimizer(a)   # 경사하강 방법을 최적화 도구로 이용
train = optimizer.minimize(cost)                   # 경사하강으로 비용이 최소화

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
