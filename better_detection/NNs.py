#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread

seed = 121
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
    hiddenlayers = []
    layers_size = [10, 10]
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
                        [last_layer_size, current_layer_size], seed=seed),
                    trainable=True)
                self.weights.append(new_w)
                log("Weight added: [{}]".format(new_w))

                #create new relation
                new_hidden = self.hiddenlayers[-1]
                new_hidden = tf.matmul(new_hidden, new_w)
                #new_hidden = tf.add(
                #    tf.matmul(new_hidden, new_w), self.biases[i - 1])
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
        x = np.reshape(x, [1, 5])
        return sess.run(
            self.hiddenlayers[-1], feed_dict={self.hiddenlayers[0]: x})

    def fit(self, x, y, sess):
        x = np.reshape(x, [1, -1])  #reshapes
        y = np.reshape(y, [1, -1])

        opt = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.cost())
        _, c = sess.run(
            [opt, self.cost()], feed_dict={self.hiddenlayers[0]: x,
                                           self.y: y})
        log(c)
        pass

    def cost(self):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=[self.hiddenlayers[-1]], labels=[self.y]))

    def activation(self, layer):
        return tf.nn.sigmoid(layer)


if (__name__ == '__main__'):
    dnn = DNN(layers=[5, 100, 100, 5], lr=0.8)
    init = tf.initialize_all_variables()

    i = [0.1, 0.2, 0.3, 0.4, 0.5]
    i = np.reshape(i, [1, -1])

    o = [0, 0, 0, 0, 0.9]
    o = np.reshape(o, [1, -1])

    with tf.Session() as sess:
        sess.run(init)

        print(dnn.query([0.1, 0.2, 0.3, 0.4, 0.5], sess))
        print(
            sess.run(dnn.cost(), feed_dict={dnn.y: o,
                                            dnn.hiddenlayers[0]: i}))
        #start training
        epoches = 100
        for e in range(epoches):
            dnn.fit(i, o, sess)
            pass

        print(dnn.query(i, sess))

        print(
            sess.run(dnn.cost(), feed_dict={dnn.y: o,
                                            dnn.hiddenlayers[0]: i}))

        dnn.weights[1] = tf.add(tf.Variable(initial_value=0.2), dnn.weights[1])

        log("changed")
        print(dnn.query(i, sess))
        print(
            sess.run(dnn.cost(), feed_dict={dnn.y: o,
                                            dnn.hiddenlayers[0]: i}))
        pass
