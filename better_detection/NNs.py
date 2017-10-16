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
                #new_hidden = tf.matmul(new_hidden, new_w) #without biases
                new_hidden = tf.add(  #with biases
                    tf.matmul(new_hidden, new_w), self.biases[i - 1])
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
        x = np.reshape(x, [1, -1])
        return sess.run(
            self.hiddenlayers[-1], feed_dict={self.hiddenlayers[0]: x})

    def fit(self, x, y, sess):
        x = np.reshape(x, [1, -1])  #reshapes
        y = np.reshape(y, [1, -1])

        opt = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr).minimize(self.cost_func())
        _, c = sess.run(
            [opt, self.cost_func()],
            feed_dict={self.hiddenlayers[0]: x,
                       self.y: y})
        log("cost: {}".format(c))
        pass

    def batch_fit(self, batch_x, batch_y):
        pass

    def batch_cost_func(self):
        pass

    def batch_cost(self, batch_x, batch_y, sess):
        pass

    def cost_func(self):
        return tf.reduce_sum(tf.square(self.hiddenlayers[-1] - self.y))
        """
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=[self.hiddenlayers[-1]], labels=[self.y]))
        """

    def cost(self, x, y, sess):
        return sess.run(
            self.cost_func, feed_dict={self.hiddenlayers[0]: x,
                                       self.y: y})

    def activation(self, layer):
        return tf.nn.sigmoid(layer)


def next_output():
    t += 1
    return [k * t]


if (__name__ == '__main__'):  #TODO: Batch
    dnn = DNN(layers=[1, 10, 10, 1], lr=1)
    init = tf.initialize_all_variables()

    i = []
    o = []
    k = 3
    x = -50

    with tf.Session() as sess:
        sess.run(init)
        
        i1=[0.2]
        i2=[0.3]
        o1=[0.6]
        o2=[0.8]
        print(tf.contrib.layers.fully_connected)
        #start training
        epoches = 100
        for e in range(epoches):
            for x in range(-10,10):
                i = [x]
                o = [x * k]
                #x += 1
                #i = np.reshape(i, [1, -1])
                #o = np.reshape(o, [1, -1])
                dnn.fit(i, o, sess)
            pass
        pass

        for i in range(-100, 100):
            i = np.reshape(i, [1, -1])
            print(dnn.query([i], sess))
        pass
