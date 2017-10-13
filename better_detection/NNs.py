#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread

seed = 129
rng = np.random.RandomState(seed)


def log(msg, urgency=-1):
    if (urgency == -1):
        print("[debug]: {}".format(msg))
    if (urgency == 0):
        print("[info]: {}".format(msg))
    if (urgency == 1):
        print("[warn]: {}".format(msg))
    if (urgency == 2):
        print("[err]: {}".format(msg))
    if (urgency == 3):
        print("[crit]: {}".format(msg))


class DNN:
    weights = []
    biases = []
    layers_size = [10, 10]
    hiddenlayers= []
    lr = 0.1
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None, None])

    def __init__(self, layers=[3, 2, 4, 5], lr=0.1):
        #layers[input_size,l1_size,l2_size,...,ln_size,output_size]

        self.input, self.layers_size, self.lr = input, layers, lr
        self.biases = tf.random_normal([layers.__len__()], seed=seed)

        last_layer_size = layers[0]
        self.hiddenlayers.append(tf.placeholder(tf.float32, [1, layers[0]]))

        for i in range(
                layers.__len__()):  #init all weight[input->hidden->output]
            if (i != 0):  #not for the input layer
                #create new weight
                current_layer_size = layers[i]
                new_w = tf.Variable(
                    tf.random_normal(
                        [last_layer_size, current_layer_size], seed=seed))
                self.weights.append(new_w)
                log("Weight added: [{}]".format(new_w))

                #create new relation
                new_hidden = self.hiddenlayers[self.hiddenlayers.__len__() - 1]
                new_hidden = tf.add(
                    tf.matmul(new_hidden, new_w),  #?
                    #tf.multiply(tf.transpose(new_hidden), new_w),  #?
                    self.biases[i - 1])
                new_hidden = self.activation(new_hidden)

                self.hiddenlayers.append(new_hidden)

                last_layer_size = current_layer_size
                pass
            pass
        self.x = tf.placeholder(tf.float32, [None, layers[0]])
        self.y = tf.placeholder(tf.float32,
                                [None, layers[layers.__len__() - 1]])

        #self.weights = np.asfarray(self.weights)

        pass

    pass

    def query(self, x, sess):
        if (x.__len__() != self.layers_size[0]):  #check input size
            raise Exception("Wrong input shape, expected length: {}".format(
                self.layers_size[0]))
            pass
        x=np.reshape(x,[1,5])
        return sess.run(self.hiddenlayers[self.hiddenlayers.__len__() - 1],feed_dict={self.hiddenlayers[0]: x})

    def fit(self):
        pass

    def activation(self, layer):
        return tf.nn.relu(layer)


if (__name__ == '__main__'):
    dnn = DNN(layers=[5, 100, 100, 5], lr=0.1)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print(dnn.query([1,2,3,4,5],sess))
        pass
