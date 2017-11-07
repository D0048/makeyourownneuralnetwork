#!/usr/bin/env python3
#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import tflearn
import numpy as np

img_h = 28
img_w = 28
img_d = 1
zdim = 200
bs = 64
lr = 0.001


def main(*argv):
    dis_input_real = tf.placeholder(
        tf.float32, shape=[bs, img_w, img_h, img_d])
    gen_input_rand = tf.placeholder(
        tf.float32, shape=[bs, img_w, img_h, img_d])

    gened_disc_input_fake = generator(gen_input_rand)
    disc_out_gened = discriminator(gened_disc_input_fake)
    disc_out_real_max = discriminator(dis_input_real, reuse=True)

    #Losses
    disc_loss = tf.reduce_mean(-disc_out_gened**2) + tf.reduce_mean(
        disc_out_real_max**2)
    gen_loss = tf.reduce_mean(disc_out_gened**2)

    #Optimizers
    opt_disc_loss = tf.train.AdamOptimizer(lr).minimize(disc_loss)
    opt_gen_min = tf.train.AdamOptimizer(lr).minimize(gen_loss)

    #Train
    with tf.Session() as sess:
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

        x_rand = tflearn.upsample_2d(x_rand, 2)

        x_rand = tflearn.upsample_2d(x_rand, 2)

        x_rand = tf.layers.conv2d(
            inputs=x_rand,
            filters=1,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.sigmoid)

        print("Generator Output: {}".format(x_rand))
        return x_rand
    pass


#Generator[-1,w,h,d]=>[-1]
def discriminator(x_img, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        print("Discriminator Input: {}".format(x_img))

        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.sigmoid)

        x_img = tf.layers.conv2d(
            inputs=x_img,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

        x_img = tf.reshape(x_img, [-1, 32 * img_h * img_w])

        x_img = tf.layers.dense(inputs=x_img, units=512, activation=tf.nn.relu)

        x_img = tf.layers.dense(
            inputs=x_img, units=1, activation=tf.nn.sigmoid)

        print("Discriminator Output: {}".format(x_img))
        return x_img
    pass


if __name__ == "__main__":
    tf.app.run()
    pass
