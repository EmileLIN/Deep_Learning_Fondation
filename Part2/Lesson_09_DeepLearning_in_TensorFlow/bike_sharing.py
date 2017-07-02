import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 


# Step 1: Load the data

data_path = '../../../datasets/Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)


# Step 2: data transformation

# use dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)


# ajust target variables

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# Step 3: Split the data into training and test data

# Save data for approximately the last 21 days as test data
test_data = data[-21*24:]

# The last data is training data
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# Take a part of training data as validation data
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# Step 4 : build the neuron network

# Set the hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 128  
display_step = 1

# initialize the weights and biases
n_features = np.shape(train_features)[1]
n_hidden_layer = 25
n_classes = np.shape(train_targets)[1]

tf.reset_default_graph()


# define the randomly initialized weights and biases
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_features, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf graph input
x = tf.placeholder("float32", [None, n_features])
y = tf.placeholder("float32", [None, n_classes])


# hidden layer 
hidden_layer = tf.add(tf.matmul(x, weights['hidden_layer']), biases['hidden_layer'])
hidden_layer = tf.sigmoid(hidden_layer, name="sigmoid")

# output_layer
output_layer  = tf.add(tf.matmul(hidden_layer, weights['out']), biases['out'])


# cost function
cost = tf.reduce_mean(tf.square(y - output_layer), name="cost")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)




# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    loss_train=[]
    loss_valid = []
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(len(train_features)/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            start = i * batch_size
            batch_x, batch_y = [train_features[start:start+batch_size], train_targets[start:start+batch_size]]
            
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # Accurancy 
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            valid = sess.run(cost, feed_dict={x: val_features, y: val_targets})

        loss_train.append(c)
        loss_valid.append(valid)

            
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "training cost=", \
                "{:.9f}".format(c))
            
            print("Epoch:", '%04d' % (epoch+1), "valid cost=", \
                "{:.9f}".format(valid))

    print("Optimization Finished!")


# make the validation plot 

#plt.plot(loss_train, label='Training loss')
#plt.plot(loss_valid, label='Validation loss')
#plt.legend()
#_ = plt.ylim()

#plt.show()


# Test data

with tf.Session() as sess:
    sess.run(init)

    predictions = sess.run(output_layer, feed_dict={x: test_features, y: test_targets})

    


plt.plot(test_targets['cnt'], label="Real values")
plt.plot(predictions[:,0], label="Predictions")
plt.legend()
_ = plt.ylim()

plt.show()



