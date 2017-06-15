#!/usr/bin/env python2.7

import numpy as np
import os
import random
from scipy import misc

data_location = '../Omniglot_data/'

# function that makes a list of all of the alphabet / character folders
def save_dir_list(data_dir, train_size=5, test_size=15, num_classes=5, Size=[105, 105]):
    path_back = "{}/Omniglot_data/images_background/".format(data_dir)
    path_eval = "{}/Omniglot_data/images_evaluation/".format(data_dir)
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

    datalist = np.asarray(datalist)

    images = []
    for dir_name in datalist:
        images.append(['{}{}'.format(dir_name, img) for img in os.listdir(dir_name) if not img.startswith('.')])
    
    images = np.asarray(images)


    #split images into train and test sets
    train_set = images[:, 0 : train_size]
    #train_set = np.reshape(train_set, len(train_set) * train_size)
    test_set = images[:, train_size : train_size + test_size]
    #test_set = np.reshape(test_set, len(test_set) * test_size)

    
    #read in the images and store
    #train set
    print('Reading in train set...')
    train_data = np.zeros([train_set.shape[0], train_set.shape[1], Size[0], Size[1]])
    for k in range(train_data.shape[0]):
        for i in range(train_data.shape[1]):
            train_data[k,i,:,:] = misc.imread(train_set[k,i])
    #test set
    print('Reading in test set...')
    test_data = np.zeros([test_set.shape[0], test_set.shape[1], Size[0], Size[1]])
    for k in range(test_data.shape[0]):
        for i in range(test_data.shape[1]):
            test_data[k,i,:,:] = misc.imread(test_set[k,i])

    np.save('{}omniglot_train'.format(data_location), train_data)
    np.save('{}omniglot_test'.format(data_location), test_data)
    return train_data, test_data


# the following code will randomly select five of these directories to use for testing and training

def get_train_data(train_data, train_size=5, test_size=15, num_classes=5, Size=[105, 105]):

    class_nums = random.sample(range(0, train_data.shape[0]), num_classes)
    train_data = train_data[class_nums]
    train_data = np.reshape(train_data, [train_size*num_classes, Size[0], Size[1]])

    train_labels = np.asarray([idx / train_size for idx in range(train_size * num_classes)])

    return train_data, train_labels

def get_test_data(test_data, train_size=5, test_size=15, num_classes=5, Size=[105,105]):

    class_nums = random.sample(range(0, test_data.shape[0]), num_classes)
    test_data = test_data[class_nums]
    test_data = np.reshape(test_data, [test_size*num_classes, Size[0], Size[1]])


    test_labels = np.asarray([idx / train_size for idx in range(test_size * num_classes)])

    return test_data, test_labels

#all_train_data, all_test_data = make_dir_list('..')
#save_dir_list('..')


all_train_data = np.load('{}omniglot_train.npy'.format(data_location))
train_set, train_labels = get_train_data(all_train_data)
all_test_data = np.load('{}omniglot_test.npy'.format(data_location))
test_set, test_labels = get_test_data(all_test_data)

print train_set.shape





