import tensorflow as tf
import threading
import numpy as np
from ops import *
import os
from PIL import Image

learning_rate = 1E-4
momentum = 0.001

batch_size = 1

num_threads = 1


def main():
    input, target = load_data()

    model = create_graph(input)

    run(model, target, input)

    # TODO
    save()

    return


def load_data():
    num_channel = 1
    # are used to feed data into our queue

    path = "./dataset/source(48)"
    path2 = "./dataset/dilate1_5(48)"
    file_list = os.listdir(path)
    file_list2 = os.listdir(path2)
    file_list = [os.path.join(path, filename) for filename in file_list]
    file_list2 = [os.path.join(path2, filename) for filename in file_list2]

    filename_queue = tf.train.string_input_producer(file_list, shuffle=False)  # list of files to read
    filename_queue2 = tf.train.string_input_producer(file_list2, shuffle=False)  # list of files to read

    dequeue_op = filename_queue.dequeue()
    dequeue_op2 = filename_queue2.dequeue()

    stacked = tf.stack([dequeue_op, dequeue_op2], axis=0)

    batched = tf.train.batch([stacked],
                             batch_size=batch_size, num_threads=1,
                             capacity=batch_size * 4)

    img = tf.map_fn(
        lambda input:
        tf.stack([tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(input[0])), tf.float32),
                  tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(input[1])), tf.float32)])
        , batched, dtype=tf.float32)
    img.set_shape([batch_size, 2, None, None, num_channel])

    return img[:, 0], img[:, 1]


def create_graph(input):
    a = pconv(input, 1, 1, 5, name="test")
    return a


def run(output, target, input):

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning_rate
    learning_rate_ = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                1000, 0.96, staircase=True) #97

    loss = tf.reduce_mean(tf.squared_difference(output, target))
    #loss = tf.nn.l2_loss(tf.subtract(output, target))
    #loss = tf.reduce_mean(tf.abs(target - output))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_)
    train = optimizer.minimize(loss, global_step=global_step)
    gradient = optimizer.compute_gradients(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)

        with tf.variable_scope("test", reuse=True):
            print(sess.run(tf.get_variable("filter")))
            print(sess.run(tf.get_variable("p")))
            print(sess.run(loss))
            pass

        for step in range(20000):
            sess.run(train)

            if (step + 1) % 100 == 0:

                with tf.variable_scope("test", reuse=True):
                    print(sess.run(tf.get_variable("filter")))
                    print(sess.run(tf.get_variable("p")))
                    pass

                l = sess.run(loss)
                print(step + 1, l)

                pass

        in_, tar, out = sess.run(
            [tf.image.convert_image_dtype(input, tf.uint8), tf.image.convert_image_dtype(target, tf.uint8),
             tf.image.convert_image_dtype(output, tf.uint8)])  # 반드시 한번에 돌릴것

        with tf.variable_scope("test", reuse=True):
            print(sess.run(tf.get_variable("filter")))
            print(sess.run(tf.get_variable("p")))
            pass

        print(sess.run(learning_rate_))
        print(out[0, :, :, :].shape)
        Image.fromarray(np.repeat(in_[0, :, :, :], 3, 2)).show() #channels = 3
        Image.fromarray(np.repeat(tar[0, :, :, :], 3, 2)).show()
        Image.fromarray(np.repeat(out[0, :, :, :], 3, 2)).show()

        coord.request_stop()
        coord.join(threads)

        return


def save():
    #TODO
    return


if __name__ == '__main__':
    main()
