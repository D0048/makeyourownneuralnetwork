#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import time
import tensorflow as tf
import tflearn
import numpy as np
import logger
import matplotlib.pyplot as plt

logger.bl_log = False
img_h = 28
img_w = 28
img_d = 1
zdim = 200
bs = 1
lr_disc = 1e-2
lr_gen = 1e-2
log_dir = "/tmp/saved_models/"


def main(*argv):
    #imshow(next_batch(5))
    #plt.waitforbuttonpress()
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
    trainable_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "Generator/*")
    trainable_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       "Discriminator/*")
    opt_disc_loss = tf.train.GradientDescentOptimizer(lr_disc).minimize(
        disc_loss, var_list=trainable_disc)
    opt_gen_loss = tf.train.GradientDescentOptimizer(lr_gen).minimize(
        gen_loss, var_list=trainable_gen)

    #GPU Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(tf.get_default_graph())

    #Train
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        imgs = []
        for i in range(2):
            img = sess.run(
                gened_fake,
                feed_dict={
                    dis_input_real: next_batch(bs),
                    gen_input_rand: np.random.uniform(
                        -1., 1., size=[bs, zdim])
                })
            for im in img:
                imgs.append(im)
        imshow(imgs)

        for e_total in range(26):
            for e_both in range(80):
                #Discriminator train
                for e in range(4):
                    _, loss = sess.run(
                        [opt_disc_loss, disc_loss],
                        feed_dict={
                            dis_input_real:
                            next_batch(bs),
                            gen_input_rand:
                            np.random.uniform(-1., 1., size=[bs, zdim])
                        })
                    print("disc_loss: {}".format(loss))
                    pass

                    #Generator train
                for e in range(30):
                    _, loss = sess.run(
                        [opt_gen_loss, gen_loss],
                        feed_dict={
                            dis_input_real:
                            next_batch(bs),
                            gen_input_rand:
                            np.random.uniform(-1., 1., size=[bs, zdim])
                        })
                    print("gen_loss: {}".format(loss))
                    pass
                pass
            pass
            #Save Model
            saver.save(sess, log_dir + "GAN_TF.ckpt")
            #Visualize
            imgs = []
            for i in range(2):
                img = sess.run(
                    gened_fake,
                    feed_dict={
                        dis_input_real: next_batch(bs),
                        gen_input_rand: np.random.uniform(
                            -1., 1., size=[bs, zdim])
                    })
                for im in img:
                    imgs.append(im)
            if (e_total % 8 == 0): imshow(imgs)
            pass
    pass


#Generator[-1,20]=>[-1,w,h,d]
def generator(x_rand, reuse=False):
    with tf.variable_scope("Generator", reuse=reuse):
        print("Generator Input: {}".format(x_rand))
        """
        x_rand = tf.layers.dense(
            inputs=x_rand,
            units=(img_h / 4) * (img_w / 4) * img_d,
            activation=tf.nn.tanh)

        x_rand = tf.reshape(
            x_rand,
            [-1, int(img_h / 4), int(img_w / 4), img_d])
        x_rand = tflearn.upsample_2d(x_rand, 2)
        x_rand = tflearn.upsample_2d(x_rand, 2)
        """
        x_rand = tf.layers.dense(
            inputs=x_rand,
            units=(img_h) * (img_w) * img_d,
            activation=tf.nn.tanh)

        x_rand = tf.reshape(x_rand, [-1, int(img_h), int(img_w), img_d])

        x_rand = tf.contrib.layers.conv2d_transpose(
            x_rand, 16, [8, 8], padding='SAME')
        x_rand = tf.contrib.layers.conv2d_transpose(
            x_rand, 16, [8, 8], padding='SAME')
        x_rand = tf.layers.conv2d(
            inputs=x_rand,
            filters=img_d,
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
        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=64,
            kernel_size=[8, 8],
            padding='same',
            activation=tf.nn.sigmoid)
        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=8,
            kernel_size=[8, 8],
            padding='same',
            activation=tf.nn.sigmoid)

        x_img = tf.reshape(x_img, [-1, 8 * img_h * img_w])
        #x_img = tf.reshape(x_img, [-1, img_d * img_h * img_w])

        x_img = tf.layers.dense(
            inputs=x_img, units=128, activation=tf.nn.sigmoid)

        x_img = tf.layers.dense(
            inputs=x_img, units=1, activation=tf.nn.sigmoid)

        print("Discriminator Output: {}".format(x_img))
        return x_img
    pass


def cli_imshow(img):
    avg = np.average(img)
    for i in img:
        for j in i:
            if (j.any() >= avg.any()):
                sys.stdout.write('█')
                pass
            else:
                sys.stdout.write(' ')
            pass
        print(" ")
        pass
    pass


def imshow(imgs):
    #f=plt.figure()
    #plt.show(block = False)
    f, a = plt.subplots(2, 10, figsize=(10, 4))
    for i in range(imgs.__len__()):
        j = 0
        print(imgs[i].shape)
        img = imgs[i].squeeze()
        #img = imgs[i]#np.asfarray([imgs[i], imgs[i], imgs[i]]).reshape([28, 28, 3])
        a[j][i].imshow(img, interpolation='nearest')
        #a[j][i].imshow(imgs[i][:, :, :], interpolation='nearest')
        pass
    f.show()
    plt.draw()
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

#"""
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
data_as_array = train_data.reshape([-1, 28, 28, 1])
print(data_as_array.shape)
imshow(data_as_array[1:5])
#"""

def next_batch(bs):
    global bs_ptr, data_as_array
    #IF OUT OF RANGE
    data_end = data_as_array.shape[0]
    bs_ptr += bs
    if (data_as_array.shape[0] < (bs_ptr + bs)):
        logger.btch_loader_log("All data cycled!")
        bs_ptr = 0
        pass
    logger.btch_loader_log('shape: {}| bs_cmp: {}'.format(
        data_as_array[bs_ptr:bs_ptr + bs, :, :, :].shape,
        data_as_array.shape[0] < (bs_ptr + bs)))
    return data_as_array[bs_ptr:bs_ptr + bs, :, :, :]

    pass


if __name__ == "__main__":
    tf.app.run()
    pass
