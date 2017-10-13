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
    hiddenlayers = []
    lr = 0.1
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
                new_hidden = self.hiddenlayers[-1]
                new_hidden = tf.add(
                    tf.matmul(new_hidden, new_w),  #?
                    #tf.multiply(tf.transpose(new_hidden), new_w),  #?
                    self.biases[i - 1])
                new_hidden = self.activation(new_hidden)

                self.hiddenlayers.append(new_hidden)

                last_layer_size = current_layer_size
                pass
            pass
        self.y = tf.placeholder(tf.float32, [None, layers[-1]])

        #self.weights = np.asfarray(self.weights)

        pass

    pass

    def query(self, x, sess):
        if (x.__len__() != self.layers_size[0]):  #check input size
            raise Exception("Wrong input shape, expected length: {}".format(
                self.layers_size[0]))
            pass
        x = np.reshape(x, [1, 5])
        return sess.run(
            self.hiddenlayers[-1], feed_dict={self.hiddenlayers[0]: x})

    def fit(self, x, y, sess):
        if (x.__len__() != self.layers_size[0]  #input check
                or y.__len__() != self.layers_size[-1]):

            raise Exception("Wrong input shape, expected length: {} \nand: {}".
                            format(self.layers_size[0], self.layers_size[-1]))
            pass

        x = np.reshape(x, [1, 5])  #reshapes
        y = np.reshape(y, [1, 5])

        opt = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr).minimize(self.cost(x))
        _, c = sess.run(
            [opt, self.cost(x)],
            feed_dict={self.hiddenlayers[0]: x,
                       self.y: y})
        log(c)
        pass

    def cost(self, x):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.hiddenlayers[-1],
                                                    labels=self.y))

    def activation(self, layer):
        return tf.nn.sigmoid(layer)


if (__name__ == '__main__'):
    dnn = DNN(layers=[5, 100, 100, 5], lr=0.1)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print(dnn.query([0.1,0.2,0.3,0.4,0.5], sess))

        epoches = 100
        for e in range(epoches):
            dnn.fit([0.1,0.2,0.3,0.4,0.5], [0.1,0.1,0.1,0.1,0.1], sess)
            pass
        print(dnn.query([0.1,0.2,0.3,0.4,0.5], sess))

        pass
