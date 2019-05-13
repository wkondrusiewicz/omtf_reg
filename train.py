import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import r2_score
import os
import time

from model import NeuralNet
from data import load_data
from params import create_parser

ep_number, batch_size, save_loc, thresh = create_parser()

#path = '../../npz_original/orig-dataset.npz'
path = '../../npz_small/small-dataset.npz'

x_train, x_val, _, y_train, y_val, _ = load_data(path, thresh=thresh)


filter_sizes = [32, 64, 128, 256]
kernel_sizes = [[6, 2], [3, 1], [3, 1], [1, 1]]
pool_sizes = [[2, 2], [3, 1], [3, 1], [1, 1]]
hidden_units = [4096, 1024, 256, 1]

net = NeuralNet(filter_sizes, kernel_sizes, pool_sizes, hidden_units)

predictions = net.create_model()

x_in, y_in, is_training = net.get_placeholders()


loss = tf.losses.mean_squared_error(
    labels=y_in, predictions=predictions)  # define loss
loss = tf.reduce_mean(loss, name='loss')  # take the mean of loss
train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)  # optimizer
init = tf.global_variables_initializer()  # initializer


saver = tf.train.Saver()

epochs = range(1, ep_number + 1)
r2_scores_tr = []
r2_scores_val = []
losses_tr = []
losses_val = []
diff_tr = np.zeros(ep_number)
diff_val = np.zeros(ep_number)
diff_tr_std = np.zeros(ep_number)
diff_val_std = np.zeros(ep_number)

with tf.Session() as sess:
    sess.run(init)  # init variables
    n = x_train.shape[0] // batch_size
    for ind, epoch in enumerate(epochs):
        # np.random.shuffle(x_train)
        # np.random.shuffle(y_train)
        tic = time.time()
        ls_tr = []
        pred_tr = []
        r2_train = []
        for i in range(n):
            start = batch_size * i
            end = batch_size * (i + 1)
            sess.run(train, feed_dict={
                y_in: y_train[start:end], x_in: x_train[start:end], is_training: True})
            #ls_val = sess.run(loss, feed_dict={y_in: y_val[start:end], x_in: x_val[start:end]})

            ls_tr_batch = sess.run(loss, feed_dict={
                y_in: y_train[start:end], x_in: x_train[start:end],  is_training: True})

            pred_tr_batch = sess.run(predictions, feed_dict={
                x_in: x_train[start:end],  is_training: True})

            r2_train_batch = r2_score(
                y_train[start:end], np.array(pred_tr_batch))

            ls_tr.append(ls_tr_batch)
            pred_tr.append(pred_tr_batch)
            r2_train.append(r2_train_batch)

        ls_tr = np.mean(ls_tr)
        pred_tr = np.mean(pred_tr)
        r2_train = np.mean(r2_train)

        pred_val = sess.run(predictions, feed_dict={
            x_in: x_val,  is_training: True})
        ls_val = sess.run(
            loss, feed_dict={y_in: y_val, x_in: x_val,  is_training: True})

        r2_val = r2_score(y_val, np.array(pred_val))

        r2_scores_tr.append(r2_train)
        r2_scores_val.append(r2_val)
        losses_tr.append(ls_tr)
        losses_val.append(ls_val)
        df_tr = (pred_tr - y_train) / y_train
        diff_tr[ind] = df_tr.mean()
        diff_tr_std[ind] = df_tr.std()

        df_val = (pred_val - y_val) / y_val
        diff_val[ind] = df_val.mean()
        diff_val_std[ind] = df_val.std()
        tac = time.time()
        print(f'\nEpoch {epoch} took {np.round(tac-tic,2)} seconds and gave following results:\nTRAIN: AVERAGED r2_score {r2_train} with loss of {ls_tr}\nVALID: r2_score {r2_val} with loss of {ls_val}')
        #print(pred_tr[:10], y_train[:10])
        #plt.hist(df + ind * 5, bins=40)
        # plt.show(block=False)
        # print(df.mean())
    save_dict = dict(zip(["r2_scores_tr", "r2_scores_val", "losses_tr", "losses_val", "diff_tr", "diff_val", "diff_tr_std", "diff_val_std"], [
                     r2_scores_tr, r2_scores_val, losses_tr, losses_val, diff_tr, diff_val, diff_tr_std, diff_val_std]))
    dir_loc = save_loc.split('/')
    dir_loc = dir_loc[:-2]
    dir_loc = '/'.join(dir_loc)
    if not os.path.exists(dir_loc):
        os.makedirs(dir_loc)
        print(dir_loc)
    saver.save(sess, save_loc)
    np.save(dir_loc + "/train.npy", save_dict)
