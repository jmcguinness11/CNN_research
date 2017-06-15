#!/usr/bin/env python2.7

import numpy as np
import os
import random
from scipy import misc

# function that makes a list of all of the alphabet / character folders
def make_dir_list(data_dir):
    path_back = "{}/omniglot/images_background/".format(data_dir)
    path_eval = "{}/omniglot/images_evaluation/".format(data_dir)
    alphadirs_back = [directory for directory in os.listdir(path_back) if not directory.startswith('.')]
    alphadirs_eval = [directory for directory in os.listdir(path_eval) if not directory.startswith('.')]

    datalist = []

    for alphabet in alphadirs_back:
        charpath = "{}{}/".format(path_back, alphabet)
        chardirs = [char for char in os.listdir(charpath) if not char.startswith('.')]
        for character in chardirs:
            datalist.append("{}{}/".format(charpath, character))

    for alphabet in alphadirs_eval:
        charpath = "{}{}/".format(path_eval, alphabet)
        chardirs = [char for char in os.listdir(charpath) if not char.startswith('.')]
        for character in chardirs:
            datalist.append("{}{}/".format(charpath, character))

    return datalist

# the following code will randomly select five of these directories to use for testing and training

def get_train_data(datalist, train_size=5, test_size=15, num_classes=5):

    class_nums = random.sample(range(0, len(datalist)), num_classes)
    datalist = np.asarray(datalist)
    dir_names = datalist[class_nums]

    images = []

    for dir_name in dir_names:
        images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

    images = np.asarray(images)

    train_set = images[:, 0 : train_size]
    train_set = np.reshape(train_set, num_classes * train_size)
    test_set = images[:, train_size : train_size + test_size]
    test_set = np.reshape(test_set, num_classes * test_size)

    test_labels = np.asarray([idx / test_size for idx in range(test_size * num_classes)])
    train_labels = np.asarray([idx / train_size for idx in range(train_size * num_classes)])

    return train_set, train_labels

def get_test_data(datalist, train_size=5, test_size=15, num_classes=5):


    class_nums = random.sample(range(0, len(datalist)), num_classes)
    datalist = np.asarray(datalist)
    dir_names = datalist[class_nums]

    images = []

    for dir_name in dir_names:
        images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])

    images = np.asarray(images)

    test_set = images[:, train_size : train_size + test_size]
    test_set = np.reshape(test_set, num_classes * test_size)

    test_labels = np.asarray([idx / test_size for idx in range(test_size * num_classes)])

    return test_set, test_labels

datalist = make_dir_list('..')
train_set, train_labels = get_train_data(datalist)
print train_set