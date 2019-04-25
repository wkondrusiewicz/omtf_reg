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
        self.x_in = tf.placeholder(tf.float32, [None, 18, 2])  # input data size
        self.y_in = tf.placeholder(tf.int32, [None])  # output data size

    def _get_conv(self, previous_layer, filter_size,  kernel_size, pool_size, last_flag=False):
        if not last_flag:
            x = tf.contrib.layers.conv2d(
                previous_layer, filter_size, kernel_size=kernel_size, padding=self.padding, activation_fn=self.activation_fn)
            x = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=self.strides)
        else:
            x = tf.contrib.layers.conv2d(
                previous_layer, filter_size, kernel_size=kernel_size, padding=self.padding, activation_fn=None)
        return x

    def _get_dense(self, previous_layer, unit, last_flag=False):
        if not last_flag:
            x = tf.layers.dense(previous_layer, unit, self.activation_fn)
        else:
            x = tf.layers.dense(previous_layer, unit, None)
        return x

    def get_placeholders(self):
        return self.x_in, self.y_in

    def create_model(self):
        x = tf.reshape(self.x_in, [-1, 18, 2, 1])
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

        x = tf.reshape(x, [-1])

        return x
