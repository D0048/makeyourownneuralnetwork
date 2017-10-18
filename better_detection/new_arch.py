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
    out_layer = tf.Variable(initial_value=0)
    activation = tf.nn.sigmoid
    optmizer = ()
    opt_min = ()
    y_label = tf.placeholder(tf.float32)
    loss = 1
    batch_y_labels = tf.placeholder(tf.float32)
    batch_x = tf.placeholder(tf.float32)

    def __init__(self, layers_size=[1, 2], lr=0.1):
        self.layers_size, self.lr = layers_size, lr
        self.activation = tf.nn.sigmoid

        last_layer_size = layers_size[1]  #layer creation
        self.layers.append(
            tf.placeholder(tf.float32, shape=[1, layers_size[0]]))

        i = 0
        for l in layers_size:
            if (i != 0):
                self.layers.append(self.new_fc_layer(self.layers[-1], l))
            i += 1
            last_layer_size = l
            pass
        self.out_layer = self.layers[-1]

        #single loss init
        self.loss = tf.losses.absolute_difference(
            labels=self.y_label, predictions=self.out_layer)
        #optimizer init
        self.optmizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr)
        self.opt_min = self.optmizer.minimize(self.loss)
        pass

    def query(self, x, sess):
        return sess.run(
            self.out_layer, feed_dict={self.layers[0]: np.reshape(x, [-1, 1])})
        pass

    def fit(self, x, y, sess):
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])
        #y=sess.run(self.out_layer,feed_dict={self.layers[0]})
        _, loss = sess.run(
            [self.opt_min, self.loss],
            feed_dict={self.layers[0]: x,
                       self.y_label: y})
        log("loss: {}".format(loss))
        pass

    def batch_fit(self, batch_x, batch_y):
        pass

    def cost(self, x, y_label, sess):
        log("loss: {}".format(
            sess.run(
                self.loss,
                feed_dict={self.layers[0]: x,
                           self.y_label: y_label})))
        pass

    def batch_cost(self):
        with tf.Session as sess:
            bc = 0
            sess.run(tf.initialize_all_variables())

            i = 0
            for x in self.batch_x:
                bc += sess.run(
                    self.cost,
                    feed_dict={
                        self.layers[0]: x,
                        self.y_label: self.batch_y_labels[i]
                    })
                i += 1
                pass
            return bc
        pass

    def new_fc_layer(
            self,
            in_layer,
            out_num, ):  #m=out_num
        log("New layer created: =>{}".format(out_num))
        layer = tf.contrib.layers.fully_connected(
            in_layer, out_num, activation_fn=self.activation)
        return layer


if (__name__ == '__main__'):
    dnn = DNN(layers_size=[1, 10, 10, 2], lr=0.1)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        i = [0.1]
        o = [0.2, 0.2]
        #performance before training
        i = np.reshape(i, [1, -1])
        print(dnn.query([i], sess))

        #start training
        import datetime
        epoches = 10
        for e in range(epoches):
            starttime = datetime.datetime.now()
            dnn.fit(i, o, sess)
            endtime = datetime.datetime.now()
            log("fitting: {}, {}, Time elapsed: {}".format(
                i, o, endtime - starttime))
            #tf.get_default_graph().finalize()
            pass
        pass

        #performance after training
        i = np.reshape(i, [1, -1])
        print(dnn.query([i], sess))
        pass
