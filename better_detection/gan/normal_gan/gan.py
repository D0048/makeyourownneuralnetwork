#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn


class MonitorCallback(tflearn.callbacks.Callback):
    itr = 0

    def __init__(self):
        self.count = 0
        pass

    def on_epoch_end(self, training_state):
        f, a = plt.subplots(2, 10, figsize=(10, 4))
        global gan
        imgs = []
        for i in range(10):
            for j in range(2):
                # Noise input.
                z = np.random.uniform(-1., 1., size=[1, z_dim])
                # Generate image from noise. Extend to 3 channels for matplot figure.
                temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
                a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
                imgs.append(np.reshape(temp, (28, 28, 3)))
                pass
            pass
        if (self.itr % 3 == 0):
            f.show()
            plt.draw()
            #imshow(imgs)
            pass
        self.itr += 1
        pass


def imshow(img):
    global X
    f, a = plt.subplots(2, 10, figsize=(10, 4))
    for i in range(10):
        j = 0
        a[j][i].imshow(img[i], interpolation='nearest')
        pass
    f.show()
    plt.draw()
    pass


from scipy import misc  # feel free to use another image loader


def load_imgs():
    return tflearn.data_utils.image_preloader(
        target_path='./Converted',
        image_shape=[256, 256],
        mode='folder',
        filter_channel=True)
    #tflearn.data_utils.build_hdf5_image_dataset(
    #    target_path='./Converted', image_shape=[256, 256], mode='folder')


# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data()
X, Y = load_imgs()

image_dim = 256 * 256  # 28*28 pixels
z_dim = 200  # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 3, 1, activation='sigmoid')
        """
        x = tflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.upsample_2d(x, 2)
        x = tflearn.conv_2d(x, 1, 5, activation='sigmoid')
        """
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        #x = tf.reshape(x, [-1, 28, 28, 1])
        x = tflearn.conv_2d(x, 64, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.conv_2d(x, 128, 5, activation='tanh')
        x = tflearn.avg_pool_2d(x, 2)
        x = tflearn.conv_2d(x, 1, 1)  #dimensionality reduction
        print(x)
        x = tf.reshape(x, [-1, 7* 7])
        x = tflearn.fully_connected(
            x, 1024,
            activation='tanh')  #64 sync with batch size.. ugly solution
        x = tflearn.fully_connected(x, 1)
        #x = tf.nn.softmax(x)
        return x


# Build Networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, 256, 256, 3], name='disc_input')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
#disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
#gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = tf.reduce_mean(disc_real)**2 - tf.reduce_mean(disc_fake)**2
gen_loss = tf.reduce_mean(disc_fake)**2

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
    op_name='GEN',
    learning_rate=0.01)
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(
    disc_real,
    placeholder=None,
    optimizer='adam',
    loss=disc_loss,
    trainable_vars=disc_vars,
    batch_size=64,
    name='target_disc',
    op_name='DISC',
    learning_rate=0.0001)
# Define GAN model, that output the generated images.
global gan
gan = tflearn.DNN(gen_model)

# Training
# Generate noise to feed to the generator
z = np.random.uniform(-1., 1., size=[total_samples, z_dim])

# Start training, feed both noise and real images.
gan.fit(
    X_inputs={gen_input: z,
              disc_input: X},
    Y_targets=None,
    n_epoch=1,
    show_metric=True,
    callbacks=[MonitorCallback()])

# Generate images from noise, using the generator network.
f, a = plt.subplots(2, 10, figsize=(10, 4))
for i in range(10):
    for j in range(2):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        # Generate image from noise. Extend to 3 channels for matplot figure.
        temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
        a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
