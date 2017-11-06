#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import glob
shuffle_data = True  # shuffle the addresses before saving
hdf5_path = './hdf5.dat'  # address to where you want to save the hdf5 file
cat_dog_train_path = './Converted/subclass_dummy/*'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]
val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]


import tables
img_dtype = tables.Float32Atom()  # dtype in which the images will be saved
# check the order of data and chose proper data shape to save images
data_shape = (0, 256, 256, 3)
# open a hdf5 file and create earrays
hdf5_file = tables.open_file('./hdf5.dat', mode='w')
train_storage = hdf5_file.create_earray(
    hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_file.create_array(hdf5_file.root, 'X', train_labels)

# a numpy array to save the mean of the images
mean = np.zeros(data_shape[1:], np.float32)
train_addrs = ''
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
    # save the image and calculate the mean so far
    train_storage.append(img[None])
    mean += img / float(len(train_labels))
    # save the image
    test_storage.append(img[None])

# save the mean and close the hdf5 file
mean_storage.append(mean[None])
hdf5_file.close()
