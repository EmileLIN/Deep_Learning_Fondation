import tensorflow as tf 



saver = tf.train.Saver()

save_file = './train_model.ckpt'

with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict({features: mnist.test.images, labels: mnist.test.labels})
    )