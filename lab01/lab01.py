import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf;


hello = tf.constant('hello tensorflow')

session = tf.Session()
print(session.run(hello))  # run hello node


n1 = tf.constant(3.0)
n2 = tf.constant(4.0)
n3 = tf.add(n1, n2)
#n3 = n1 + n2

print(n1, n2)
print(n3)


print(session.run([n1, n2]))
print(session.run(n3))

# b'hello tensorflow'
# Tensor("Const_1:0", shape=(), dtype=float32) Tensor("Const_2:0", shape=(), dtype=float32)
# Tensor("Add:0", shape=(), dtype=float32)
# [3.0, 4.0]
# 7.0


# placeholder

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b

print(session.run(add_node, feed_dict={a: 3, b: 5}))
print(session.run(add_node, feed_dict={a: [1, 2, 3], b: [2, 3, 4]}))

# 8.0
# [3. 5. 7.]
