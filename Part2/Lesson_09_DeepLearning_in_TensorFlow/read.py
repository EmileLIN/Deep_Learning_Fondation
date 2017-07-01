import tensorflow as tf 

#Remove the previous zeights and bias
tf.reset_default_graph()

save_file = './model.ckpt'

#Two variables
weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the zeights and bias
    saver.restore(sess, save_file)

    # Show the values of zeights and bias
    print('Weights:')
    print(sess.run(weights))

    print('Bias')
    print(sess.run(bias))
