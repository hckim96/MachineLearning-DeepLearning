import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x_train = [2, 4, 7]
y_train = [4, 9, 19]

w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x_train * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # 편차 제곱의 평균


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for step in range(3001):
    session.run(train)
    if step % 30 == 0:
        print(step, session.run(cost), session.run(w), session.run(b))


# using place holder

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w2 = tf.Variable(tf.random_normal([1]), name="weight")
b2 = tf.Variable(tf.random_normal([1]), name="bias")


hypothesis2 = X * w2 + b2
cost2 = tf.reduce_mean(tf.square(hypothesis2 - Y))

optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train2 = optimizer2.minimize(cost2)
session.run(tf.global_variables_initializer())

for step in range(3001):

    c_v, w_v, b_v, t_v = session.run([cost2, w2, b2, train2], feed_dict={
        X: [1, 2, 3], Y: [3, 5, 7]})
    if step % 30 == 0:
        print(step, c_v, w_v, b_v, t_v)


# feed_dict 로 만들어진 모델에 대해서 값을 따로 넘겨줄수있다


# test model
print(session.run(hypothesis2, feed_dict={X: [1, 2, 3, 4, 5, 6]}))
