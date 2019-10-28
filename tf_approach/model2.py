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
        self.is_training = tf.placeholder(
            tf.bool, [], name='is_training')  # training indicator
        self.batch_size = None
        self.shuffle_subset_size = None

    def _get_conv(self, previous_layer, filter_size,  kernel_size, pool_size, last_flag=False):
        x = tf.contrib.layers.conv2d(
        previous_layer, filter_size, kernel_size=kernel_size, padding=self.padding, activation_fn=None)
        if not last_flag:
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(
                inputs=x, pool_size=pool_size, strides=self.strides)
        return x

    def _get_dense(self, previous_layer, unit, last_flag=False):
        x = tf.layers.dense(previous_layer, unit, None)
        if not last_flag:
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.dropout(
                x, keep_prob=0.5, is_training=self.is_training)
        return x

    def get_placeholders(self):
        return self.x_in, self.y_in, self.is_training

    def set_basic_params(self, shuffle_subset_size, batch_size):
        self.batch_size = batch_size
        self.shuffle_subset_size= shuffle_subset_size

    def _set_dataset(self):
        assert self.batch_size is not None, 'Please set batch_size using set_basic_params()'
        assert self.shuffle_subset_size is not None, 'Please set thresh using set_basic_params()'

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_in, self.y_in))
        train_dataset = train_dataset.shuffle(self.shuffle_subset_size)
        train_dataset = train_dataset.batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_in, self.y_in))
        val_dataset = val_dataset.shuffle(self.shuffle_subset_size)
        val_dataset = val_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_in, self.y_in))
        test_dataset = test_dataset.shuffle(self.shuffle_subset_size)
        test_dataset = test_dataset.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        features, labels = iterator.get_next()

        train_iterator = iterator.make_initializer(
            train_dataset, name='train_iterator')
        val_iterator = iterator.make_initializer(
            val_dataset, name='val_iterator')
        test_iterator = iterator.make_initializer(
            test_dataset, name='test_iterator')

        # dataset = dataset.shuffle(self.thresh)
        # dataset = dataset.batch(self.batch_size)
        # dataset = dataset.batch(self.batch_size).repeat()
        # iter = train_dataset.make_initializable_iterator()
        # data_iter_init = iter.make_initializer(dataset, name='dataset_iter')

        return features, labels, train_iterator, val_iterator, test_dataset

    def create_model(self):
        # iter = self._set_dataset()
        # features, labels= iter.get_next()
        features, labels, train_iterator, val_iterator, test_dataset = self._set_dataset()
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

        labels = tf.identity(labels, name="labels")

        return x, labels, train_iterator, val_iterator, test_dataset
