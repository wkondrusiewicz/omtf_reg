import tensorflow as tf
import json
import numpy as np
from data import load_data
import matplotlib.pyplot as plt

# with open('basic_info.json', 'r') as f:
#     data = json.load(f)
#     thresh = data["thresh"]

thresh = 50000
batch_size = 128
EPOCHS = 5

path = '../../npz_small/small-dataset.npz'

x_train, x_val, _, y_train, y_val, _ = load_data(path, thresh=thresh)

print(type(x_train))

x_in = tf.placeholder(tf.float32, [None, 18, 2])  # input data size
y_in = tf.placeholder(tf.int32, [None])  # output data size

dataset = tf.data.Dataset.from_tensor_slices(
    (x_in, y_in))
dataset = dataset.shuffle(thresh)
dataset = dataset.batch(batch_size).repeat()

plt.hist(y_train)
plt.savefig('batches/orig.png')

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
features = tf.reshape(features, [-1, 36])

# make a simple model
# pass the first value from iter.get_next() as input
net = tf.layers.dense(features, 8, activation=tf.tanh)
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
prediction = tf.reshape(prediction, [-1])
print(prediction.shape, labels.shape, y_in.shape)

loss = tf.losses.mean_squared_error(
    labels=labels, predictions=prediction)

# loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    print('Training...')
    n_batches = thresh//batch_size
    for i in range(EPOCHS):
        sess.run(iter.initializer, feed_dict={ x_in: x_train, y_in: y_train})
        tot_loss = 0
        count = 0
        chuj= []
        # print(np.sort(y_train))
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
            temp = list(sess.run(labels))
            count += len(temp)
            # print(temp)
            chuj = chuj + temp
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
        # print(count)
        # chuj.sort()
        # print(chuj)
        plt.hist(chuj)
        plt.savefig('batches/'+str(i)+'.png')
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict={ x_in: x_val, y_in: y_val})
    print('Test Loss: {:4f}'.format(sess.run(loss)))
