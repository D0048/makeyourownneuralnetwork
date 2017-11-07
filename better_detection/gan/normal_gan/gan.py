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


def img_loader_init():
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./Converted/subclass_dummy/*"))
    image_reader = tf.WholeFileReader()
    return image_reader, filename_queue


def next_img(reader, quene):
    # filename which we are ignoring.
    _, image_file = reader.read(quene)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    image = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
    return image


# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data()
#X, Y = load_imgs()

image_dim = 28 * 28  # 28*28 pixels
z_dim = 200  # Noise data points
total_samples = 998  # len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 4)
        #x = tflearn.upsample_2d(x, 2)
        x = tf.cast(x, tf.float32)
        x = tflearn.conv_2d(x, 3, 1, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        #x = tf.reshape(x, [-1, 28, 28, 1])
        #x = tflearn.conv_2d(x, 32, 5, activation='tanh')
        #x = tflearn.avg_pool_2d(x, 2)
        #x = tflearn.conv_2d(x, 128, 5, activation='tanh')
        #x = tflearn.avg_pool_2d(x, 2)
        #x = tflearn.conv_2d(x, 1, 1)  #dimensionality reduction

        x = tf.reshape(x, [-1, 28 * 28])
        x = tflearn.fully_connected(x, 256, activation='sigmoid')
        x = tflearn.fully_connected(x, 1)
        return x


# Build Networks
gen_input = tflearn.input_data(
    shape=[None, z_dim], name='input_noise', dtype=tf.float32)
disc_input = tflearn.input_data(
    shape=[None, 28, 28, 3], name='disc_input', dtype=tf.float32)

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_loss = tf.reduce_mean(disc_real)**2. - tf.reduce_mean(disc_fake)**2.
gen_loss = tf.reduce_mean(disc_fake)**2.

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
    batch_size=1,
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
    batch_size=1,
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
loader, quene = img_loader_init()
"""
X, Y = tflearn.data_utils.image_preloader(
    "./Converted/",
    image_shape=(28, 28),
    mode='folder',
    normalize=True,
    filter_channel=True)
print("X: {}\nY: {}".format(X,Y))
"""
print(next_img(loader, quene).eval())
for i in range(1):
    with tf.Session() as sess:
        gan.fit(
            X_inputs={
                gen_input: z,
                disc_input: next_img(loader, quene).eval()
            },
            Y_targets=None,
            n_epoch=3,
            show_metric=False,
            callbacks=[MonitorCallback()])
        pass
    pass

# Generate images from noise, using the generator network.
f, a = plt.subplots(2, 10, figsize=(10, 4))
for i in range(10):
    for j in range(2):
        z = np.random.uniform(-1., 1., size=[1, z_dim])

        #temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
        #a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
        img = gan.predict([z]).reshape((28, 28, 3))
        a[j][i].imshow(img, interpolation='nearest')
f.show()
plt.draw()
plt.waitforbuttonpress()
