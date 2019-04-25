import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import r2_score

#from model import create_model, create_placeholders

from model import NeuralNet
from data import load_data
from params import create_parser

ep_number, batch_size, plottable = create_parser()

path = '../npz_small/small-dataset.npz'

x_train, x_test, y_train, y_test = load_data(path, 200)



filter_sizes = [32, 64, 128, 256]
kernel_sizes = [[6, 2], [3, 1], [3, 1], [1, 1]]
pool_sizes = [[2,2],[3,1],[3,1],[1,1]]
hidden_units = [4096,1024,256,1]

#net = NeuralNet(filter_sizes=filter_sizes, kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, hidden_units=hidden_units)
net = NeuralNet(filter_sizes, kernel_sizes, pool_sizes, hidden_units)

predictions = net.create_model()
"""
x_in, y_in = create_placeholders()

predictions = create_model(x_in=x_in)
"""
x_in, y_in = net.get_placeholders()


loss = tf.losses.mean_squared_error(
    labels=y_in, predictions=predictions)  # define loss
loss = tf.reduce_mean(loss)  # take the mean of loss
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)  # optimizer
init = tf.global_variables_initializer()  # initializer


epochs = range(1, ep_number + 1)
train_scores = []
test_scores = []

with tf.Session() as sess:
    sess.run(init)  # init variables
    n = x_train.shape[0] // batch_size
    for epoch in epochs:
        # np.random.shuffle(x_train)
        # np.random.shuffle(y_train)
        for i in range(n):
            start = batch_size * i
            end = batch_size * (i + 1)
            ls_tr, _ = sess.run([loss, train], feed_dict={
                                y_in: y_train[start:end], x_in: x_train[start:end]})

        pred_tr = sess.run(predictions, feed_dict={x_in: x_train[:n]})
        r2_train = r2_score(y_train[:n], np.array(pred_tr).astype(int))
        print(f'Epoch {epoch} gave r2_score {r2_train} with loss of {ls_tr}')
        train_scores.append(r2_train)
    ls_te = sess.run(loss, feed_dict={y_in: y_test[:n], x_in: x_test[:n]})
    pred_te = sess.run(predictions, feed_dict={x_in: x_test[:n]})
    print(
        f'Test gave r2_score {r2_score(y_test[:n], np.array(pred_te).astype(int))} with loss of {ls_te}')
if plottable:
    plt.plot(epochs, train_scores)
    plt.title(f'r2 score for training for {ep_number} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('r2 score')
    plt.show()
