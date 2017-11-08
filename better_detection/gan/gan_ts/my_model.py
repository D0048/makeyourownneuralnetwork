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


def main(*argv):
    dis_input_real = tf.placeholder(
        tf.float32, shape=[bs, img_w, img_h, img_d])
    gen_input_rand = tf.placeholder(tf.float32, shape=[bs, zdim])

    gened_disc_input_fake = generator(gen_input_rand)
    disc_out_gened = discriminator(gened_disc_input_fake)
    disc_out_real_max = discriminator(dis_input_real, reuse=True)

    #Losses
    #"""Classic
    disc_loss = tf.log(tf.reduce_mean(disc_out_gened)) - tf.log(
        tf.reduce_mean(disc_out_real_max))
    gen_loss = -tf.log(tf.reduce_mean(disc_out_gened))
    """
    disc_loss = (tf.reduce_mean(disc_out_gened)
                 )**2 - tf.reduce_mean(disc_out_real_max)**2
    gen_loss = -(tf.reduce_mean(disc_out_gened))**2
    """

    #Optimizers
    opt_disc_loss = tf.train.AdamOptimizer(lr_disc).minimize(disc_loss)
    opt_gen_loss = tf.train.AdamOptimizer(lr_gen).minimize(gen_loss)

    #Train
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        print(
            sess.run(
                gened_disc_input_fake,
                feed_dict={
                    dis_input_real: next_batch(bs),
                    gen_input_rand: np.random.uniform(
                        -1., 1., size=[bs, zdim])
                }))
        for e_total in range(10):
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
        #TODO: wrong output size

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
    bs_ptr += bs
    return data_as_array[bs_ptr:bs_ptr + bs, :, :, :]
    pass


if __name__ == "__main__":
    tf.app.run()
    pass
