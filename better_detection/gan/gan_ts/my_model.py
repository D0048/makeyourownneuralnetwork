#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import tflearn
import numpy as np

img_h = 64
img_w = 64
img_d = 3
zdim = 200
bs = 5
lr_disc = 1e-5
lr_gen = 1e-5
log_dir = "/tmp/saved_models/"


def main(*argv):
    dis_input_real = tf.placeholder(
        tf.float32, shape=[bs, img_w, img_h, img_d])
    gen_input_rand = tf.placeholder(tf.float32, shape=[bs, zdim])

    gened_fake = generator(gen_input_rand)
    disc_fake = discriminator(gened_fake)
    disc_real = discriminator(dis_input_real, reuse=True)

    #Losses
    #"""Classic
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))

    #Optimizers
    opt_disc_loss = tf.train.AdamOptimizer(lr_disc).minimize(disc_loss)
    opt_gen_loss = tf.train.AdamOptimizer(lr_gen).minimize(gen_loss)

    #GPU Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(tf.get_default_graph())

    #Train
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Test Generation: {}".format(
            sess.run(
                gened_fake,
                feed_dict={
                    dis_input_real: next_batch(bs),
                    gen_input_rand: np.random.uniform(
                        -1., 1., size=[bs, zdim])
                })))

        for e_total in range(100):
            #Discriminator train
            for e in range(5):
                _, loss = sess.run(
                    [opt_disc_loss, disc_loss],
                    feed_dict={
                        dis_input_real: next_batch(bs),
                        gen_input_rand: np.random.uniform(
                            -1., 1., size=[bs, zdim])
                    })
                print("disc_loss: {}".format(loss))
                pass

                #Generator train
            for e in range(5):
                _, loss = sess.run(
                    [opt_gen_loss, gen_loss],
                    feed_dict={
                        dis_input_real: next_batch(bs),
                        gen_input_rand: np.random.uniform(
                            -1., 1., size=[bs, zdim])
                    })
                print("gen_loss: {}".format(loss))
                pass

            #Save Model
            saver.save(sess, log_dir + "GAN_TF.ckpt")
            pass
        pass
    pass


#Generator[-1,20]=>[-1,w,h,d]
def generator(x_rand, reuse=False):
    with tf.variable_scope("Generator", reuse=reuse):
        print("Generator Input: {}".format(x_rand))

        x_rand = tf.layers.dense(
            inputs=x_rand,
            units=(img_h / 4) * (img_w / 4) * img_d,
            activation=tf.nn.relu)

        x_rand = tf.reshape(
            x_rand,
            [-1, int(img_h / 8), int(img_w / 8), img_d])

        #x_rand = tflearn.upsample_2d(x_rand, 2)
        x_rand = tflearn.upsample_2d(x_rand, 2)
        x_rand = tflearn.upsample_2d(x_rand, 2)

        x_rand = tf.layers.conv2d(
            inputs=x_rand,
            filters=3,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.sigmoid)

        print("Generator Output: {}".format(x_rand))
        return x_rand
    pass


#Discriminator[-1,w,h,d]=>[-1]
def discriminator(x_img, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        print("Discriminator Input: {}".format(x_img))
        """
        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.sigmoid)
        """
        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=6,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)

        x_img = tf.reshape(x_img, [-1, 6 * img_h * img_w])
        #x_img = tf.reshape(x_img, [-1, img_d * img_h * img_w])

        x_img = tf.layers.dense(inputs=x_img, units=128, activation=tf.nn.relu)

        x_img = tf.layers.dense(
            inputs=x_img, units=1, activation=tf.nn.sigmoid)

        print("Discriminator Output: {}".format(x_img))
        return x_img
    pass


import numpy, h5py
global bs_ptr, data_as_array
bs_ptr = 0
f = h5py.File('./dataset.hdf5', 'r')
data = f.get('train_img')
data_as_array = numpy.array(data)

#Rescale
import scipy
buf = []
for img in data_as_array:
    buf.append(scipy.misc.imresize(img, (img_h, img_w)))
    pass
data_as_array = np.asfarray(buf)


def next_batch(bs):
    global bs_ptr, data_as_array
    #IF OUT OF RANGE
    data_end = data_as_array.shape[0]
    if (data_as_array[bs_ptr:bs_ptr + bs, :, :, :].shape[0] < bs):#TODO
        print("All data cycled!")
        bs_ptr = 0
        pass
    bs_ptr += bs
    print('shape: {}\n bs_cmp: {}'.format(
        data_as_array[bs_ptr:bs_ptr + bs, :, :, :].shape,
        data_as_array[bs_ptr:bs_ptr + bs, :, :, :].shape[0] < bs))
    return data_as_array[bs_ptr:bs_ptr + bs, :, :, :]

    pass


if __name__ == "__main__":
    tf.app.run()
    pass
