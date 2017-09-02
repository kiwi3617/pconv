import tensorflow as tf
import numpy as np
import os
from PIL import Image

path = "../dataset/source(48)"
file_list = os.listdir(path)
file_list = [os.path.join(path, filename) for filename in file_list]

filename_queue = tf.train.string_input_producer(file_list)  # list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_image(value)  # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):  # length of your filename list
        image = my_img.eval()  # here is your image Tensor :)
        image = np.asarray(image)
        if len(image.shape) == 4:
            image = image.squeeze(0)
        print(image.shape)
        Image.fromarray(np.repeat(image,3,2)).show()

        image = image[np.newaxis, ...]
        #image = np.clip(image,100,256)
        image[np.where(image == 0)] = 1
        print(image)
        #image, np.finfo(np.float32).eps

    coord.request_stop()
    coord.join(threads)

input = tf.image.convert_image_dtype(image, tf.float32)

#p = tf.random_normal([])
p = tf.constant(-10,dtype=tf.float32)

f = tf.constant(np.reshape(np.diag(np.ones([11])*1),[11,11,1,1]), dtype=tf.float32) #(11, 11, 1, 1)

p1 = tf.pow(input, p + 1)
p2 = tf.pow(input, p)
c1 = tf.nn.conv2d(p1, f, strides=[1, 1, 1, 1], padding="SAME")
c2 = tf.nn.conv2d(p2, f, strides=[1, 1, 1, 1], padding="SAME")
pconv = tf.divide(c1, c2)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(pconv))

out = sess.run(tf.image.convert_image_dtype(pconv, tf.uint8))
out = out.squeeze(0)

pp1 = sess.run(tf.image.convert_image_dtype(p1, tf.uint8))
pp1 = pp1.squeeze(0)

pp2 = sess.run(tf.image.convert_image_dtype(p2, tf.uint8))
pp2 = pp2.squeeze(0)

cc1 = sess.run(tf.image.convert_image_dtype(c1, tf.uint8))
cc1 = cc1.squeeze(0)


cc2 = sess.run(tf.image.convert_image_dtype(c2, tf.uint8))
cc2 = cc2.squeeze(0)

print(sess.run(p))

Image.fromarray(np.repeat(pp1,3,2)).show()
Image.fromarray(np.repeat(pp2,3,2)).show()
Image.fromarray(np.repeat(cc1,3,2)).show()
Image.fromarray(np.repeat(cc2,3,2)).show()
Image.fromarray(np.repeat(out,3,2)).show()