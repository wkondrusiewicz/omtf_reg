import tensorflow as tf

class NeuralNet():
    def __init__(self, filter_sizes, kernel_sizes, pool_sizes, hidden_units, strides=2, activation_fn=tf.nn.relu):
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.hidden_units = hidden_units
        self.strides = strides
        self.padding = "same"
        self.activation_fn = activation_fn
        self.x_in = tf.placeholder(
            tf.float32, [None, 18, 2], name='InData')  # input data size
        self.y_in = tf.placeholder(
            tf.int32, [None], name='OutData')  # output data size
        self.is_training = tf.placeholder(tf.bool, [], name = 'is_training') #training indicator
        self.batch_size=None
        self.thresh = None
    def _get_conv(self, previous_layer, filter_size,  kernel_size, pool_size, last_flag=False):
        if not last_flag:
            x = tf.contrib.layers.conv2d(
                previous_layer, filter_size, kernel_size=kernel_size, padding=self.padding, activation_fn=self.activation_fn)
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=pool_size, strides=self.strides)
        else:
            x = tf.contrib.layers.conv2d(
                previous_layer, filter_size, kernel_size=kernel_size, padding=self.padding, activation_fn=None)
        return x

    def _get_dense(self, previous_layer, unit, last_flag=False):
        if not last_flag:
            x = tf.layers.dense(previous_layer, unit, self.activation_fn)
            x = tf.contrib.layers.dropout(x, keep_prob = 0.25, is_training=self.is_training)
        else:
            x = tf.layers.dense(previous_layer, unit, None)
        return x

    def get_placeholders(self):
        return self.x_in, self.y_in, self.is_training

    def set_basic_params(self, thresh, batch_size):
        self.batch_size=batch_size
        self.thresh = thresh

    def _set_dataset(self):
        assert self.batch_size is not None, 'Please set batch_size using set_basic_params()'
        assert self.thresh is not None, 'Please set thresh using set_basic_params()'

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_in, self.y_in))
        dataset = dataset.shuffle(self.thresh)
        dataset = dataset.batch(self.batch_size).repeat()
        iter = dataset.make_initializable_iterator()
        return iter

    def create_model(self):
        iter = self._set_dataset()
        features, labels= iter.get_next()
        x = tf.reshape(features, [-1, 18, 2, 1])
        last_flag = False
        for i, (f, k, p) in enumerate(zip(self.filter_sizes, self.kernel_sizes, self.pool_sizes)):
            if i == len(self.filter_sizes) - 1:
                last_flag = True
            x = self._get_conv(x, f, k, p, last_flag)

        x = tf.reshape(x, [-1, 1 * 1 * self.filter_sizes[-1]])
        last_flag = False
        for i, unit in enumerate(self.hidden_units):
            if i == len(self.hidden_units) - 1:
                last_flag = True
            x = self._get_dense(x, unit, last_flag)

        x = tf.reshape(x, [-1], name="predictions")

        return x, labels, iter
