#!/usr/bin/env python3
#from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn


class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self):
        self.count = 0
        pass

    def on_epoch_end(self, training_state):
        global gan
        z_dim = 784
        f, a = plt.subplots(2, 10, figsize=(10, 4))
        for i in range(10):
            for j in range(2):
                # Noise input.
                #z = np.random.uniform(-1., 1., size=[1, z_dim])
                # Generate image from noise. Extend to 3 channels for matplot figure.
                global z
                temp = [[ii, ii, ii] for ii in list(gan.predict(z)[0])]
                a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
                pass
            pass
        if (self.count % 10 == 0):
            f.show()
            plt.draw()
            plt.waitforbuttonpress()
            pass
        self.count += 1
        pass


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, 64, activation='relu')
        x = tflearn.fully_connected(x, 10, activation='sigmoid')
        return x


# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data()
image_dim = 784  # 28*28 pixels
z_dim = 200  # Noise data points
total_samples = len(X)

Y = tf.one_hot(Y, 10,dtype=tf.float32)
with tf.Session() as sess:
    Y=sess.run(Y)
    pass

# Build Networks
#gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
gen_input = tflearn.input_data(shape=[None, 784], name='disc_input')
disc_input = tflearn.input_data(shape=[None, 784], name='disc_input')
Y_feed = tflearn.input_data(shape=[None, 10], name='Y_feed')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Lossdisc_input
disc_loss = tf.losses.sigmoid_cross_entropy(disc_real, Y_feed)
gen_loss = tf.losses.sigmoid_cross_entropy(disc_fake, Y_feed)
"""
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(
    tf.log(disc_fake)) + tf.constant(0.5) * tf.reduce_sum(
        (gen_input - generator(gen_input, reuse=True))**2)
"""

# Build Training Ops for both Generator and Discriminator.
# Each network optimization should only update its own variable, thus we need
# to retrieve each network variables (with get_layer_variables_by_scope) and set
# 'placeholder=None' because we do not need to feed any target.
gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gen_model = tflearn.regression(
    gen_sample,
    placeholder=None,
    optimizer='adam',
    loss=gen_loss,
    trainable_vars=gen_vars,
    batch_size=64,
    name='target_gen',
    op_name='GEN')
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(
    disc_real,
    placeholder=None,
    optimizer='adam',
    loss=disc_loss,
    trainable_vars=disc_vars,
    batch_size=64,
    name='target_disc',
    op_name='DISC')
# Define GAN model, that output the generated images.
global gan
gan = tflearn.DNN(gen_model)

# Training
# Generate noise to feed to the generator
z_dim = 784
z = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Start training, feed both noise and real images.

gan.fit(
    X_inputs={  #gen_input: z,
        gen_input: X,
        disc_input: X,
        Y_feed: Y
    },
    Y_targets=None,
    n_epoch=50,
    callbacks=[MonitorCallback()])
