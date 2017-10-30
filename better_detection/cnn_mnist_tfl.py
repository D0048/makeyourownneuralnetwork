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

from tflearn.datasets import mnist

#aug rot25|32,3|mxp2|64,3|64,3|mxp2|fc512|drpot0.5|fc10|->adam0.001 =>99.5 0.01
#aug rot25|32,5|128,5|64,5|mxp2|drpot0.5|16,4|20,5|16,5|fc1024|fc1024|fc1024|fc10|->adam0.001 =>99.43 0.027
#aug !|32,5|128,5|64,5|mxp2|drpot0.5|16,4|20,5|16,5|fc1024|fc1024|fc1024|fc10|->adam0.001 =>99.1 0.020


def main(unused_argv):
    #global rec_model, old_graph, new_graph
    #rec_model = train_rec()

    X, Y, X_test, Y_test = mnist.load_data(one_hot=True)

    img_prep = ImagePreprocessing()
    img_aug = ImageAugmentation()

    inputs = input_data(
        shape=[None, 28, 28, 1],
        data_preprocessing=img_prep,
        data_augmentation=img_aug)
    # Classifier
    classifier_fc1 = fully_connected(
        inputs, 128, activation='relu', name="clssfr_fc1")
    classifier_fc2 = fully_connected(
        inputs, 128, activation='relu', name="clssfr_fc2")
    classifier_fc3 = fully_connected(
        inputs, 10, activation='relu', name="clssfr_fc3")
    classifier_out = regression(
        classifier_fc3, optimizer='adam', loss=new_cost, learning_rate=0.001)
    print(classifier_out)
    exit()

    # Confuser
    confuser_fc1 = fully_connected(inputs, 128, activation='relu')
    confuser_fc2 = fully_connected(confuser_fc1, 128, activation='relu')
    confuser_fc3 = fully_connected(confuser_fc2, 28 * 28, activation='relu')
    confuser_out = regression(
        confuser_fc3, optimizer='adam', loss=new_cost, learning_rate=0.001)

    new_model = tflearn.DNN(new_network, tensorboard_verbose=0)
    new_model.fit(
        X,
        Y,
        n_epoch=5,
        shuffle=False,
        validation_set=(X_test, Y_test),
        show_metric=True,
        batch_size=1,
        run_id='mnist_counter')

    pass


def new_cost(y_pred, y_true):
    with tf.name_scope(None):
        return tf.losses.sigmoid_cross_entropy(y_true, y_pred)

    pass


def train_rec():
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
    inputs = input_data(
        shape=[None, 28, 28, 1],
        data_preprocessing=img_prep,
        data_augmentation=img_aug,
        name="inputs")
    network = conv_2d(inputs, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)

    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.6)
    network = fully_connected(network, 10, activation='relu')
    network = regression(
        network,
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate=0.001)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    global rec_input, rec_network
    rec_input, rec_network = inputs, network
    """
    model.fit(
        X,
        Y,
        n_epoch=1,
        shuffle=True,
        validation_set=(X_test, Y_test),
        show_metric=True,
        batch_size=128,
        run_id='mnist')
    """
    return model


def exit():
    import sys
    sys.exit(0)
    pass


if __name__ == "__main__":
    tf.app.run()
