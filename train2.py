import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import r2_score
import os
import time

from model2 import NeuralNet
from data import load_data
from params import create_parser

ep_number, batch_size, thresh, path_model, path_data = create_parser()

path = '../../npz_original/orig-dataset.npz'
# path = '../../npz_small/small-dataset.npz'

x_train, x_val, _, y_train, y_val, _ = load_data(path_data, thresh=thresh)

if not os.path.exists(path_model):
    os.makedirs(path_model)
else:
    print('Model save location already exists')


print(y_train.shape, y_val.shape)


filter_sizes = [32, 64, 128, 256]
kernel_sizes = [[6, 2], [3, 1], [3, 1], [1, 1]]
pool_sizes = [[2, 2], [3, 1], [3, 1], [1, 1]]
hidden_units = [4096, 1024, 256, 1]

net = NeuralNet(filter_sizes, kernel_sizes, pool_sizes, hidden_units)

x_in, y_in, is_training = net.get_placeholders()

shuffle_subset_size = x_train.shape[0] #we have to ensure that it is greater or equal than size os training set, for val and test will be bigger that their sizes

net.set_basic_params(shuffle_subset_size, batch_size)

predictions, labels, train_iterator, val_iterator, _ = net.create_model()


loss = tf.losses.mean_squared_error(
    labels=labels, predictions=predictions)  # define loss
loss = tf.reduce_mean(loss, name='loss')  # take the mean of loss


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # to make batch_norm work properly
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)  # optimizer

init = tf.global_variables_initializer()  # initializern


saver = tf.train.Saver()

epochs = range(1, ep_number + 1)
r2_scores_tr = np.zeros(ep_number)
r2_scores_val = np.zeros(ep_number)
losses_tr = np.zeros(ep_number)
losses_val = np.zeros(ep_number)
diff_tr = np.zeros(ep_number)
diff_val = np.zeros(ep_number)
diff_tr_std = np.zeros(ep_number)
diff_val_std = np.zeros(ep_number)

with tf.Session() as sess:
    sess.run(init)  # init variables
    n_tr = 1+x_train.shape[0] // batch_size
    n_val = 1+x_val.shape[0] // batch_size
    for ind, epoch in enumerate(epochs):
        # print("TRAIN\n")
        tic = time.time()

        sess.run(train_iterator, feed_dict={x_in: x_train, y_in: y_train})
        tot_loss = 0
        pred_tr = np.zeros(0)
        labels_tr = np.zeros(0)
        losses = np.zeros(n_tr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            for i in range(n_tr):
                _, loss_value, pred, lab = sess.run(
                    [train_op, loss, predictions, labels], feed_dict={is_training: True})
                tot_loss += loss_value
                pred_tr = np.concatenate([pred_tr, pred])
                labels_tr = np.concatenate([labels_tr, lab])
                losses[i] = loss_value
                # print(f"i={i} pred.shape={np.array(pred).shape}")

        # print("VALID\n")


        losses_tr[ind] = losses.mean()
        r2_scores_tr[ind] = r2_score(labels_tr, pred_tr)
        diff_tr[ind] = np.mean((pred_tr - labels_tr) / labels_tr)
        diff_tr_std[ind] = np.std((pred_tr - labels_tr) / labels_tr)


        sess.run(val_iterator, feed_dict={x_in: x_val, y_in: y_val})
        tot_loss = 0

        pred_val = np.zeros(0)
        labels_val = np.zeros(0)
        losses = np.zeros(n_val)
        for i in range(n_val):
            # _, loss_value, pred, lab = sess.run([train_op, loss, predictions, labels], feed_dict={is_training: True})
            loss_value, pred, lab = sess.run([loss, predictions, labels], feed_dict={is_training: False})
            tot_loss += loss_value
            pred_val = np.concatenate([pred_val, pred])
            labels_val = np.concatenate([labels_val, lab])
            losses[i] = loss_value
            # print(f"i={i} pred.shape={np.array(pred).shape}")

        print(labels_val[:10], pred_val[:10])

        losses_val[ind] = losses.mean()
        r2_scores_val[ind] = r2_score(labels_val, pred_val)
        diff_val[ind] = np.mean((pred_val - labels_val) / labels_val)
        diff_val_std[ind] = np.std((pred_val - labels_val) / labels_val)

        tac = time.time()

        print(f'\nEpoch {epoch} took {np.round(tac-tic,2)} seconds and gave following results:\nTRAIN: r2_score {r2_scores_tr[ind]} with loss of {losses_tr[ind]}\nVALID: r2_score {r2_scores_val[ind]} with loss of {losses_val[ind]}')

    save_dict = dict(zip(["r2_scores_tr", "r2_scores_val", "losses_tr", "losses_val", "diff_tr", "diff_val", "diff_tr_std", "diff_val_std"], [
                     r2_scores_tr, r2_scores_val, losses_tr, losses_val, diff_tr, diff_val, diff_tr_std, diff_val_std]))

    saver.save(sess, path_model)
    np.save(os.path.dirname(path_model) +  "/train.npy", save_dict)
