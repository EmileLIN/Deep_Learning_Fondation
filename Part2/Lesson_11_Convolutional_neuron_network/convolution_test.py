import tensorflow as tf 

# Output depth
k_output = 64

# Image properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter 
filter_size_width = 5
filter_size_height = 5

# Input/ Image
input = tf.placeholder(tf.float32, shape=[None, image_height, image_width, color_channel])

# Weight and bias
weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))


# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1,2,2,1], padding='SAME')

# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)

# add activation function
conv_layer = tf.nn.relu(conv_layer)

# max pooling
conv_layer = tf.nn.max_pool(conv_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


