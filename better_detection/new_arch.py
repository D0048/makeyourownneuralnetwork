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

    layers = []
    layers_size = [1]
    lr = 0.1

    def __init__(self, layers_size=[1, 2], lr=0.1):
        self.layers_size, self.lr = layers_size, lr

        last_layer_size = layers_size[1]
        self.layers.append(
            tf.placeholder(tf.float32, shape=[1, layers_size[0]]))

        i = 0
        for l in layers_size:
            if (i != 0):
                self.layers.append(
                    self.new_fc_layer(self.layers[-1], last_layer_size, l))
            i += 1
            last_layer_size = l
        pass

    def query(self, x, sess):
        pass

    def fit(self, x, y, sess):
        pass

    def batch_fit(self, batch_x, batch_y):
        pass

    def cost_func(self):
        pass

    def cost(self, x, y, sess):
        pass

    def batch_cost_func(self):
        pass

    def batch_cost(self, batch_x, batch_y, sess):
        pass

    def activation(self, layer):
        pass

    def new_fc_layer(self, in_layer, in_num, out_num, use_relu=False):  #m=out_num
        log("New layer created: {}=>{}".format(in_num, out_num))
        weights = tf.Variable(
            tf.truncated_normal([in_num, out_num], stddev=0.05))
        biases = tf.Variable(tf.truncated_normal(shape=[out_num], stddev=0.05))

        layer = tf.add(tf.matmul(in_layer, weights), biases)
        if use_relu:
            layer = tf.nn.relu(layer)
        else:
            layer = tf.nn.sigmoid(layer)
        return layer


if (__name__ == '__main__'):  #TODO: Batch
    dnn = DNN(layers_size=[1, 10, 10, 1], lr=1)
    SystemExit()
    init = tf.initialize_all_variables()

    i = []
    o = []
    k = 3
    x = -50

    with tf.Session() as sess:
        sess.run(init)
        i1 = [0.2]
        i2 = [0.3]
        o1 = [0.6]
        o2 = [0.8]
        print(tf.contrib.layers.fully_connected)

        #performance before training
        for i in range(-5, 5):
            i = np.reshape(i, [1, -1])
            print(dnn.query([i], sess))
        pass

        #start training
        import datetime
        epoches = 10
        for e in range(epoches):
            for x in range(-10, 10):
                starttime = datetime.datetime.now()
                i = [x]
                o = [x * k]
                #i = np.reshape(i, [1, -1])
                #o = np.reshape(o, [1, -1])
                dnn.fit(i, o, sess)
                endtime = datetime.datetime.now()
                log("fitting: {}, {}, Time elapsed: {}".format(
                    i, o, endtime - starttime))
                #tf.get_default_graph().finalize()
            pass
        pass

        #performance after training
        for i in range(-5, 5):
            i = np.reshape(i, [1, -1])
            print(dnn.query([i], sess))
        pass
