import tensorflow as tf
from data import load_data
import json
import numpy as np
from sklearn.metrics import r2_score

with open('basic_info.json','r') as f:
    save_loc = json.load(f)["save_loc"]

path = '../npz_small/small-dataset.npz'

x_train, x_test, y_train, y_test = load_data(path,1000)

test_batch_size = 512

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(save_loc + '.meta')
    new_saver.restore(sess, save_loc)
    n = x_test.shape[0] // test_batch_size
    losses = []
    r2_scores = []
    print(f'Starting test:\n')
    for i in range(n):
        start = test_batch_size * i
        end = test_batch_size * (i + 1)
        pr, l = sess.run(['predictions:0', 'loss:0'], feed_dict={
                         "InData:0": x_test[start:end], 'OutData:0': y_test[start:end]})
        r2 = r2_score(y_test[start:end], np.array(pr))
        losses.append(l)
        r2_scores.append(r2)
    print(f'Test gave averaged: r2_score = {np.mean(r2_scores)} with loss = {np.mean(losses)}')
