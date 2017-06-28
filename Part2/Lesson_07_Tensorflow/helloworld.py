import tensorflow as tf

hello_constant = tf.constant("Hello, Carla")



# tf holder

x = tf.placeholder(tf.string)

# basic computation

y = tf.add(5,2)

z = tf.subtract(10,4)

w = tf.multiply(2 ,5)


# type convert

a = tf.subtract(tf.cast(tf.constant(2.0), tf.int32) ,tf.constant(1))


with tf.Session() as sess:

	output = sess.run(hello_constant)
	print(output)

	output = sess.run(x, feed_dict={x: 'Hello World'})
	print(output)


with tf.Session() as sess2:

	output = sess2.run(y)
	print(output)

	output = sess2.run(a)
	print(output)


# basic computation


w = tf.divide(tf.constant(10), tf.constant(2))
z = tf.subtract(tf.cast(w, tf.int32), tf.constant(1))

with tf.Session() as sess3:
	output = sess3.run(z)
	print(output)



#update weight

#initialize all variables
n_features = 120
n_labels = 5
weights = tr.Variable(tf.truncated_normal((n_features, n_labels)))

n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)






















