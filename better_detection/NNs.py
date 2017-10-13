#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread

seed = 128
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
    weights = [[[]]]

    def __init__(self, hidden=[10, 10], input=784,
                 lr=0.1):  #hidden[l1_size,l2_size,...,ln_size]
        assert (hidden, [[int]])
        last_layer_size = input
        for i in range(hidden.__len__()):
            current_layer_size = hidden[i]
            self.weights.append(
                tf.Variable(
                    tf.random_normal(
                        [last_layer_size, current_layer_size], seed=seed)))
            log("Weight added: [{},{}]".format(last_layer_size,
                                               current_layer_size))
            last_layer_size = current_layer_size
        pass

    def fit(self):
        pass

    def query(self):
        pass


if (__name__ == '__main__'):
    dnn = DNN()
