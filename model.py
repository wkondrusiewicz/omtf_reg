import tensorflow as tf


def create_placeholders():
    x_in = tf.placeholder(tf.float32, [None, 18, 2])
    y_in = tf.placeholder(tf.int32, [None])

    return x_in, y_in


def create_model(x_in):
    x = tf.reshape(x_in, [-1, 18, 2, 1])

    x = tf.contrib.layers.conv2d(
        x, 32, kernel_size=[6, 2], padding="same", activation_fn=tf.nn.relu)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)

    x = tf.contrib.layers.conv2d(
        x, 64, kernel_size=[3, 1], padding="same", activation_fn=tf.nn.relu)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[3, 1], strides=2)

    x = tf.contrib.layers.conv2d(
        x, 128, kernel_size=[3, 1], padding="same", activation_fn=tf.nn.relu)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[3, 1], strides=2)

    x = tf.contrib.layers.conv2d(
        x, 256, kernel_size=[1, 1], padding="same", activation_fn=tf.nn.relu)

    x = tf.reshape(x, [-1, 1 * 1 * 256])

    x = tf.layers.dense(x, 4096, tf.nn.relu)
    x = tf.layers.dense(x, 1024, tf.nn.relu)
    x = tf.layers.dense(x, 256, tf.nn.relu)
    x = tf.layers.dense(x, 1, None)
    x = tf.reshape(x, [-1])

    return x
