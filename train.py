import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import r2_score
import seaborn as sns
import os

from model import NeuralNet
from data import load_data
from params import create_parser

ep_number, batch_size, plottable, save_loc, thresh = create_parser()

path = '../../npz_small/small-dataset.npz'

x_train, x_test, y_train, y_test = load_data(path, thresh=thresh)

filter_sizes = [32, 64, 128, 256]
kernel_sizes = [[6, 2], [3, 1], [3, 1], [1, 1]]
pool_sizes = [[2, 2], [3, 1], [3, 1], [1, 1]]
hidden_units = [4096, 1024, 256, 1]

net = NeuralNet(filter_sizes, kernel_sizes, pool_sizes, hidden_units)

predictions = net.create_model()

x_in, y_in = net.get_placeholders()


loss = tf.losses.mean_squared_error(
    labels=y_in, predictions=predictions)  # define loss
loss = tf.reduce_mean(loss, name='loss')  # take the mean of loss
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)  # optimizer
init = tf.global_variables_initializer()  # initializer


saver = tf.train.Saver()

epochs = range(1, ep_number + 1)
r2_scores = []
losses = []
diff = np.zeros(ep_number)
diff_std = np.zeros(ep_number)

with tf.Session() as sess:
    sess.run(init)  # init variables
    n = x_train.shape[0] // batch_size
    for ind, epoch in enumerate(epochs):
        # np.random.shuffle(x_train)
        # np.random.shuffle(y_train)
        for i in range(n):
            start = batch_size * i
            end = batch_size * (i + 1)
            ls_tr, _ = sess.run([loss, train], feed_dict={
                                y_in: y_train[start:end], x_in: x_train[start:end]})

        pred_tr = sess.run(predictions, feed_dict={x_in: x_train})
        r2_train = r2_score(y_train, np.array(pred_tr))
        print(f'Epoch {epoch} gave r2_score {r2_train} with loss of {ls_tr}')
        r2_scores.append(r2_train)
        losses.append(ls_tr)
        df = (pred_tr - y_train) / y_train
        diff[ind] = df.mean()
        diff_std[ind] = df.std()


        #print(pred_tr[:10], y_train[:10])
        plt.hist(df+ind*5,bins=40)
        plt.show(block=False)
        #print(df.mean())

    dir_loc = save_loc.split('/')
    dir_loc = dir_loc[:-2]
    dir_loc = '/'.join(dir_loc)
    if not os.path.exists(dir_loc):
        os.makedirs(dir_loc)
    saver.save(sess, save_loc)

if plottable:
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, r2_scores)
    plt.title(f'r2 score for training for {ep_number} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('r2 score')

    plt.subplot(3, 1, 2)
    plt.plot(epochs, losses)
    plt.title(f'Loss for training for {ep_number} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(3, 1, 3)
    plt.plot(epochs, diff)
    #plt.fill_between(epochs, diff + diff_std, diff - diff_std, alpha=0.1)

    plt.title(f'Loss for training for {ep_number} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Pull')

    plt.tight_layout()
    plt.show()
