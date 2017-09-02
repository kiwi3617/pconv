import tensorflow as tf
import numpy as np

eps = tf.constant(1.0E-12, dtype=tf.float32)


def conv(batch_input, out_channels, stride, filter_size=4, name="conv"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        in_channels = batch_input.get_shape()[3]

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))

        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")

        return conv


def pconv(batch_input, out_channels, stride, filter_size=4, name="pconv"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        in_channels = batch_input.get_shape()[3]
        strides = [1, stride, stride, 1]
        eps = tf.constant(1.0E-7, dtype=tf.float32)

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 trainable=False,
                                 initializer=tf.random_uniform_initializer(0, 1))
                                 # initializer=tf.constant_initializer(np.diag(np.ones(5))))
        p = tf.get_variable("p", [], dtype=tf.float32, trainable=True,
                            initializer=tf.random_normal_initializer(0, 0.02))

        batch_input = batch_input+eps

        powered1 = tf.pow(batch_input, p + 1) # 0^-1 !!!
        powered2 = tf.pow(batch_input, p)

        conv1 = tf.nn.conv2d(powered1, filter, strides, padding="SAME")
        conv2 = tf.nn.conv2d(powered2, filter, strides, padding="SAME")

        divided = tf.divide(conv1, conv2 + eps)

        return divided


def sconv(batch_input, out_channels, stride, filter_size=4, name="sconv"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        in_channels = batch_input.get_shape()[3]
        strides = [1, stride, stride, 1]

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 trainable=True,
                                 # initializer=tf.random_normal_initializer(0.5, 0))
                                 # initializer = tf.random_uniform_initializer(0.0, 0.0))
                                 initializer=tf.constant_initializer(np.diag(np.ones(5))))
        p = tf.get_variable("p", [], dtype=tf.float32, trainable=True,
                            initializer=tf.random_normal_initializer(6, 0.00))

        multiplied = tf.multiply(p, batch_input)
        powered = tf.exp(multiplied)
        multipled_powered_input = tf.multiply(batch_input, powered)

        conv1 = tf.nn.conv2d(multipled_powered_input, filter, strides, padding="SAME")
        conv2 = tf.nn.conv2d(powered, filter, strides, padding="SAME")
        divided = tf.divide(conv1, conv2 + eps)

        return divided


def lseconv(batch_input, out_channels, stride, filter_size=4, name="lseconv"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        in_channels = batch_input.get_shape()[3]
        strides = [1, stride, stride, 1]

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 trainable=True,
                                 # initializer=tf.random_normal_initializer(0.5, 0))
                                 # initializer = tf.random_uniform_initializer(0.0, 0.0))
                                 initializer=tf.constant_initializer(np.diag(np.ones(5))))
        p = tf.get_variable("p", [], dtype=tf.float32, trainable=True,
                            initializer=tf.random_normal_initializer(10, 0.00))

        multiplied = tf.multiply(batch_input, p)
        exp = tf.exp(multiplied)
        filter = filter / (tf.reduce_sum(filter) + eps)
        conv = tf.nn.conv2d(exp, filter, strides, padding="SAME")
        log = tf.log(conv)
        # average = tf.nn.avg_pool(log, [1, filter_size, filter_size, 1], strides, padding="SAME")
        divided = tf.divide(log, p)  # 1/0 !!!

        return divided


def gconv(batch_input, out_channels, stride, filter_size=4, name="gconv"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        in_channels = batch_input.get_shape()[3]
        strides = [1, stride, stride, 1]

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 trainable=True,
                                 # initializer=tf.random_normal_initializer(0.5, 0))
                                 # initializer = tf.random_uniform_initializer(0.0, 0.0))
                                 initializer=tf.constant_initializer(np.diag(np.ones(5))))
        p = tf.get_variable("p", [], dtype=tf.float32, trainable=True,
                            initializer=tf.random_normal_initializer(1, 0.00)
                            )

        powered = tf.pow(batch_input, p)
        filter = filter / (tf.reduce_sum(filter) + eps)  # sum(filter) == 1
        conv = tf.nn.conv2d(powered, filter, strides, padding="SAME")
        gconv = tf.pow(conv, 1 / p)  # 1/0 !!!

        # tf.cond(tf.equal(p, 0),, powered_average)

        return gconv


def dilate(batch_input, out_channels, stride, filter_size=4, name="dilate"):
    with tf.variable_scope(name):
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]

        in_channels = batch_input.get_shape()[3]
        strides = [1, stride, stride, 1]
        rates = [1, 1, 1, 1]

        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels], dtype=tf.float32,
                                 trainable=True,
                                 #                         initializer=tf.random_normal_initializer(0.5, 0))
                                 #                         initializer = tf.random_uniform_initializer(0.0, 0.0))
                                 initializer=tf.constant_initializer(np.diag(np.ones(5))))

        dilate = tf.nn.dilation2d(batch_input, filter, strides, rates, padding="SAME")
        return dilate
