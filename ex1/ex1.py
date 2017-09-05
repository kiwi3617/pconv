import tensorflow as tf
import threading
import numpy as np
from ops import *
import os
from PIL import Image


learning_rate = 5E-4
momentum = 0.001
batch_size = 1
num_threads = 1


def main():

#    with tf.name_scope('load_data'):
    input, target = load_data()

    model = create_graph(input)

    tf.summary.histogram(name='model_histo', values=model)

    run(model, target, input, False)

    # TODO
    save()

    return


def load_data():
    num_channel = 1
    # are used to feed data into our queue

    path = '../dataset/source(48)'
    path2 = '../dataset/dilate1_5(48)'
    file_list = os.listdir(path)
    file_list2 = os.listdir(path2)
    file_list = [os.path.join(path, filename) for filename in file_list]
    file_list2 = [os.path.join(path2, filename) for filename in file_list2]

    with tf.name_scope('push_source'):
        filename_queue = tf.train.string_input_producer(file_list, shuffle=False)  # list of files to read
        dequeue_op = filename_queue.dequeue()
    with tf.name_scope('push_target'):
        filename_queue2 = tf.train.string_input_producer(file_list2, shuffle=False)  # list of files to read
        dequeue_op2 = filename_queue2.dequeue()

    with tf.name_scope('stack_proc'):
        stacked = tf.stack([dequeue_op, dequeue_op2], axis=0)

    with tf.name_scope('batch_proc'):
        batched = tf.train.batch([stacked],
                                 batch_size=batch_size, num_threads=1,
                                 capacity=batch_size * 4)
    with tf.name_scope('return_each'):
        img = tf.map_fn(
            lambda input:
            tf.stack([tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(input[0])), tf.float32),
                      tf.image.convert_image_dtype(tf.image.decode_png(tf.read_file(input[1])), tf.float32)])
            , batched, dtype=tf.float32,name='__source_target__')
        img.set_shape([batch_size, 2, None, None, num_channel])

    #tf.summary.histogram(name='batch_img', values=img)

    return img[:, 0], img[:, 1]


def create_graph(input):
    a = sconv(input, 1, 1, 5, name="Main_Graph")
    return a


def run(output, target, input, restore):
    saver = tf.train.Saver()

    if (restore == False):
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope('learning_rate'):
            starter_learning_rate = learning_rate
            learning_rate_ = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                        1000, 0.96, staircase=True,name='decay_learning_rate')  # 97

        tf.summary.scalar(name='decay_learning_rate', tensor=learning_rate_)

        with tf.name_scope('cost_function'):
            loss = tf.reduce_mean(tf.squared_difference(output, target,name='sqr_diff'),name='mse')
            # loss = tf.nn.l2_loss(tf.subtract(output, target))
            # loss = tf.reduce_mean(tf.abs(target - output))


        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_)

        train = optimizer.minimize(loss, global_step=global_step)
        #gradient = optimizer.compute_gradients(loss)
        tf.summary.scalar(name='loss', tensor=loss)


        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(init)

            with tf.variable_scope("Main_Graph", reuse=True):
                print("filter :", sess.run(tf.get_variable("filter")))
                print("p value : ",sess.run(tf.get_variable("p")))
                print("loss : ",sess.run(loss))
                pass

            img_difference = sess.run(target-output)
            tf.summary.histogram(name='tgt_opt,histo',values=img_difference)
            tf.summary.histogram(name='source_histo', values=input)
            tf.summary.histogram(name='target_histo', values=target)
            tf.summary.histogram(name='output_histo', values=output)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logdir='./board/sample5', graph=sess.graph)

            for step in range(50000):
                sess.run(train)

                if (step) % 1000 == 0:
                    with tf.variable_scope("Main_Graph", reuse=True):
                        #filter_summary = \
                        sess.run(tf.get_variable('filter'))
                        #p_summary = \
                        sess.run(tf.get_variable('p'))
                        #print("filter :" ,filter_summary)
                        #print("p value:" , p_summary)

                    l = sess.run(loss)
                    result = sess.run([merged])
                    train_writer.add_summary(result[0], step)
                    print("step :",step, l)




            in_, tar, out = sess.run(
                [tf.image.convert_image_dtype(input, tf.uint8,name='source_img'),
                tf.image.convert_image_dtype(target, tf.uint8,name='target_img'),
                tf.image.convert_image_dtype(output, tf.uint8,name='output_img')])  # 반드시 한번에 돌릴것

            with tf.variable_scope("Main_Graph", reuse=True):
                print("filter : ",sess.run(tf.get_variable("filter")))
                print("p value : ",sess.run(tf.get_variable("p")))

            print("decay_learning_rate",sess.run(learning_rate_))
            print("shape",out[0, :, :, :].shape)
            Image.fromarray(np.repeat(in_[0, :, :, :], 3, 2)).show() #channels = 3
            Image.fromarray(np.repeat(tar[0, :, :, :], 3, 2)).show()
            Image.fromarray(np.repeat(out[0, :, :, :], 3, 2)).show()



            coord.request_stop()
            coord.join(threads)


            #saver.save(sess,'./ckpt/sconv_model',global_step=2000)
    else:
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(init)

            print(sess.run(tf.get_variable("filter")))
            print(sess.run(tf.get_variable("p")))


            in_, tar, out = sess.run(
                [tf.image.convert_image_dtype(input, tf.uint8,name='source_img'),
                tf.image.convert_image_dtype(target, tf.uint8,name='target_img'),
                tf.image.convert_image_dtype(output, tf.uint8,name='output_img')])  # 반드시 한번에 돌릴것

            with tf.variable_scope("Main_Graph", reuse=True):
                print(sess.run(tf.get_variable("filter")))
                print(sess.run(tf.get_variable("p")))

            print(out[0, :, :, :].shape)
            Image.fromarray(np.repeat(in_[0, :, :, :], 3, 2)).show() #channels = 3
            Image.fromarray(np.repeat(tar[0, :, :, :], 3, 2)).show()
            Image.fromarray(np.repeat(out[0, :, :, :], 3, 2)).show()


            coord.request_stop()
            coord.join(threads)
        return



def save():
    # TODO
    return


if __name__ == '__main__':
    main()

