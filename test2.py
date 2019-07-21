import tensorflow as tf
from data import load_data
import json
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from model2 import NeuralNet
import os
pt_intervals = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12,
                14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]


def eff_curve(p_req, intervals):
    res = np.zeros(len(intervals))
    for i, c in enumerate(intervals):
        res[i] = len(p_req[p_req > c])
    return res / len(p_req)


with open('basic_info.json', 'r') as f:
    data = json.load(f)
    thresh = data["thresh"]
    path_model = data['path_model']
    path_data = data['path_data']
#path = '../../npz_original/orig-dataset.npz'
path = '../../npz_small/small-dataset.npz'

_, _, x_test, _, _, y_test = load_data(path_data, thresh=thresh)

test_batch_size = 256

# y_in =tf.placeholder(
#     tf.int32, [None], name='OutData')
# x_in = tf.placeholder(
#     tf.float32, [None, 18, 2], name='InData')
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     (x_in, y_in))
# dataset = dataset.shuffle(thresh)
# dataset = dataset.batch(test_batch_size).repeat()
# iter = dataset.make_initializable_iterator()
# features, labels= iter.get_next()
init = tf.global_variables_initializer()  # initializer

with tf.Session() as sess:
    sess.run(init)
    print(f'{x_test.shape[0]}')
    new_saver = tf.train.import_meta_graph(path_model + '.meta')
    new_saver.restore(sess, path_model)

    graph = tf.get_default_graph()

    # hehe = [n.name for n in graph.as_graph_def().node]
    # print(hehe)
    # Restore the init operation
    # iter = graph.get_operation_by_name('dataset_iter')
    # _,_ = iter.get_next()
    n_te = 1+x_test.shape[0] // test_batch_size
    losses = []
    r2_scores = []
    print(f'\nStarting test:')
    eff = np.zeros((n_te, len(pt_intervals)))
    # sess.run('dataset_iter', feed_dict={"InData:0": x_test, 'OutData:0': y_test})
    sess.run('test_iterator', feed_dict={"InData:0": x_test, 'OutData:0': y_test})


    tot_loss = 0

    pred_te = np.zeros(0)
    labels_te = np.zeros(0)
    losses = np.zeros(n_te)
    for i in range(n_te):
        loss_value, pred, lab = sess.run(["loss:0", "predictions:0", "labels:0"], feed_dict={"is_training:0": False})
        tot_loss += loss_value
        pred_te = np.concatenate([pred_te, pred])
        labels_te = np.concatenate([labels_te, lab])
        losses[i] = loss_value
        eff[i] = eff_curve(pred, pt_intervals)

    losses = losses.mean()
    r2_scores_te = r2_score(labels_te, pred_te)

    # losses = []
    # for i in range(n_te):
    #     start = test_batch_size * i
    #     end = test_batch_size * (i + 1)
    #     pr, l, lab = sess.run(['predictions:0', 'loss:0','labels:0'], feed_dict={'is_training:0': False})
    #     r2 = r2_score(y_test[start:end], np.array(pr))
    #
    #     print(f"i={i} pred.shape={np.array(pr).shape}")
    #
    #     losses.append(l)
    #     r2_scores.append(r2)
    #     eff[i] = eff_curve(pr, pt_intervals)
    #     print(pr[:10], l, lab[:10])
    # # print(labels_te[:10], pred_te[:10])
    #
    # print(np.mean(losses))

    print(
        f'Test gave averaged: r2_score = {r2_scores_te} with loss = {losses}\n')
    save_dict = {"pt_intervals": pt_intervals, "eff": eff}
    np.save(os.path.dirname(path_model)+'/test.npy', save_dict)
