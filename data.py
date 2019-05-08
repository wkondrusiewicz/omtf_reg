import numpy as np


def load_data(path, thresh=None):
    data = np.load(path, 'r')

    train_data = data['TRAIN']
    test_data = data['TEST']
    tr_data = train_data[()]
    te_data = test_data[()]

    #y_train = tr_data['PT_CODE']
    y_train = tr_data['PT_VAL']
    x_train = tr_data['HITS']
    #y_test = te_data['PT_CODE']
    y_test = te_data['PT_VAL']
    x_test = te_data['HITS']

    x_train = x_train[y_train<=100]
    y_train = y_train[y_train<=100]
    x_test = x_test[y_test<=100]
    y_test = y_test[y_test<=100]

    if thresh is not None:
        x_train = x_train[:thresh]
        y_train = y_train[:thresh]

        x_test = x_test[:thresh]
        y_test = y_test[:thresh]

    return x_train, x_test, y_train, y_test
