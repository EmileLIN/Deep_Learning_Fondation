import numpy as np
import tensorflow as tf


# hyperparams

num_epochs = 100
truncated_backprop_length = 15
total_series_length = 50000
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


# Step 1 Collect data

def generateData():
    x = np.array(np.random.choice(2, total_series_length,p=[0.5,0.5])) 

    y = np.roll(x, echo_step)

    y[0:echo_step] = 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x,y)

data = generateData()

print(data)


# step2 build the model

batchX_placeholder = tf.placeholder(tf.float32, truncated_backprop_length)
batchY_placeholder = tf.placeholder(tf.float32, truncated_backprop_length)

init_state = tf.placeholder(tf.placeholder(tf.float32, [batch_size, state_size]))


W = tf.Variable(np.random.rand(state_size+1, state_size), dtype = tf.float32)
b = tf.Variable(np.zeros(1, state_size), dtype = tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes))
b2 = tf.Variable(np.zeros(1, num_classes), dtype = tf.float32)

# unpack matrix into 1 dimension array
input_series = tf.unpack(batchX, placeholder, axis =1)
label_series = tf.unpack(batchY_placeholder, axis =1)

 

 # Forward pass
 current_state = init_state
 state_series = []

 for current_input in input_series:
     current_input = tf.reshape(current_input,[batch_size, 1])

     input_and_state_concatented = tf.concat(1,[current_input, current_state])

     next_state = tf.tanh(tf.matmul(input_and_state_concatented, W) + b)

     states_series.append(next_state)

     current_state = next_state

# calculate loss and minimize it 
logits_series = [tf.matmul[state, W2]+b2 for state in state_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series, label_series)]

total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2,3,1)

    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series]
        single_output_series = np.array([(1 if out[0] <0.5 else 0) for out in ])

        plt.subplot(2,3, batch_series_idx+2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0 ,2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :]* 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width =1, color="green")

    plt.draw()
    plt.pause(0.0001)       



# Step 3   Traning the network
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    plt.ion()
    plt.figure()
    plt.show()
    loss_list= []

    for epoch_idx in range(num_epochs):
        x, y = generateData()

        #init state
        _current_state = np.zeros((batch_size, state_size))

        print("New Data , epoch", epoch_idx)

        for batch_idx = in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start + truncated_backprop_length

            batchX = X[:, start_idx, end_idx]
            batchY = y[:, start_idx, end_idx]

            _total_loss = _train_step, _current_state, _prediction_series = sess.run([total_loss, train_step, current_state, predictions_series],
                feed_dict={ batchX_placeholder: batchX, batchY_placeholder:batchY, init_state:_current_state})

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _prediction_series, batchX, batchY)

plt.ioff()
plt.show()


