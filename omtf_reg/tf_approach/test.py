import tensorflow as tf
from data import load_data
import json
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


pt_intervals = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12,
                14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]


def eff_curve(p_req, intervals):
    res = np.zeros(len(intervals))
    for i, c in enumerate(intervals):
        res[i] = len(p_req[p_req > c])
    return res / len(p_req)


with open('basic_info.json', 'r') as f:
    data = json.load(f)
    save_loc = data["save_loc"]
    thresh = data["thresh"]

#path = '../../npz_original/orig-dataset.npz'
path = '../../npz_small/small-dataset.npz'

_, _, x_test, _, _, y_test = load_data(path, thresh=thresh)

test_batch_size = 512

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(save_loc + '.meta')
    new_saver.restore(sess, save_loc)
    n = x_test.shape[0] // test_batch_size
    losses = []
    r2_scores = []
    print(f'\nStarting test:')
    eff = np.zeros((n, len(pt_intervals)))
    for i in range(n):
        start = test_batch_size * i
        end = test_batch_size * (i + 1)
        pr, l = sess.run(['predictions:0', 'loss:0'], feed_dict={
                         "InData:0": x_test[start:end], 'OutData:0': y_test[start:end],  'is_training:0': False})
        r2 = r2_score(y_test[start:end], np.array(pr))
        losses.append(l)
        r2_scores.append(r2)
        eff[i] = eff_curve(pr, pt_intervals)
    print(
        f'Test gave averaged: r2_score = {np.mean(r2_scores)} with loss = {np.mean(losses)}\n')
    save_dict = {"pt_intervals": pt_intervals, "eff": eff}
    dir_loc = save_loc.split('/')
    dir_loc = dir_loc[:-2]
    dir_loc = '/'.join(dir_loc)
    np.save(dir_loc+'/test.npy', save_dict)
