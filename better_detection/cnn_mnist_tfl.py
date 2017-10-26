#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

tf.logging.set_verbosity(tf.logging.INFO)

#aug rot25|32,3|mxp2|64,3|64,3|mxp2|fc512|drpot0.5|fc10|->adam0.001 =>99.5 0.01
#aug rot25|32,5|128,5|64,5|mxp2|drpot0.5|16,4|20,5|16,5|fc1024|fc1024|fc1024|fc10|->adam0.001 =>99.43 0.027
#aug !|32,5|128,5|64,5|mxp2|drpot0.5|16,4|20,5|16,5|fc1024|fc1024|fc1024|fc10|->adam0.001 =>99.1 0.020


def main(unused_argv):
    from tflearn.datasets import mnist
    X, Y, X_test, Y_test = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    X_test = X_test.reshape([-1, 28, 28, 1])
    #X, Y = shuffle(X, Y)
    #Y = to_categorical(Y,10)
    #Y_test = to_categorical(Y_test,10)

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    #img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(
        shape=[None, 28, 28, 1],
        data_preprocessing=img_prep,
        data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 512, activation='relu')
    print(network)
    import sys
    sys.exit(0)
    tf.truncated_normal([-1, 512])  #[batch_size, weight_per_self]
    network = tf.matmul()
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation='relu')
    network = regression(
        network,
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate=0.001)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(
        X,
        Y,
        n_epoch=50,
        shuffle=True,
        validation_set=(X_test, Y_test),
        show_metric=True,
        batch_size=96,
        run_id='mnist')
    pass


if __name__ == "__main__":
    tf.app.run()
