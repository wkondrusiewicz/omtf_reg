import numpy as np


def load_data(path, thresh):
    data = np.load(path, 'r')

    train_data = data['TRAIN']
    test_data = data['TEST']
    tr_data = train_data[()]
    te_data = test_data[()]

    y_train = tr_data['PT_CODE']
    x_train = tr_data['HITS']
    y_test = te_data['PT_CODE']
    x_test = te_data['HITS']

    x_train = x_train[:thresh]
    y_train = y_train[:thresh]

    x_test = x_test[:thresh]
    y_test = y_test[:thresh]
    return x_train, x_test, y_train, y_test
