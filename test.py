import tensorflow as tf
from data import load_data
import json
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def eff_curve(p_req, codes):
    res = np.zeros(len(codes))
    for i, c in enumerate(codes):
        res[i] = len(p_req[p_req > c]) / len(p_req)
    return res


with open('basic_info.json', 'r') as f:
    data = json.load(f)
    save_loc = data["save_loc"]
    thresh = data["thresh"]

path = '../../npz_small/small-dataset.npz'

_, x_test, _, y_test = load_data(path, thresh=thresh)

test_batch_size = 512

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(save_loc + '.meta')
    new_saver.restore(sess, save_loc)
    n = x_test.shape[0] // test_batch_size
    losses = []
    r2_scores = []
    print(f'Starting test:\n')
    eff = np.zeros((n, 30))
    for i in range(n):
        start = test_batch_size * i
        end = test_batch_size * (i + 1)
        pr, l = sess.run(['predictions:0', 'loss:0'], feed_dict={
                         "InData:0": x_test[start:end], 'OutData:0': y_test[start:end]})
        r2 = r2_score(y_test[start:end], np.array(pr))
        losses.append(l)
        r2_scores.append(r2)
        eff[i] = eff_curve(pr, range(30))
    print(
        f'Test gave averaged: r2_score = {np.mean(r2_scores)} with loss = {np.mean(losses)}')


plt.plot(range(30), eff.mean(axis=0))
plt.title(f'Effectivness curve')
plt.xlabel('$p_T$ codes')
plt.ylabel('Effectivness')
plt.tight_layout()

plt.show()
