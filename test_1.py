from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
from tqdm import tqdm

n_test_image = 28
time = 18

# Load data
X_train = np.load('D:/Bitcamp/Project/Frontalization/Numpy/color_128_x.npy') # Side face
Y_train = np.load('D:/Bitcamp/Project/Frontalization/Numpy/color_128_y.npy') # Front face

# print(X_train.shape)
# print(Y_train.shape)

# X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size = 0.064, shuffle = True, random_state = 66)

X_test = np.load('D:/Bitcamp/Project/Frontalization/Numpy/lsm_x.npy') # Side face
# Y_test = np.load('‪D:/Bitcamp/Project/Frontalization/Numpy/lsm_y.npy') # Front face
Y_test_path = 'D:/Bitcamp/Project/Frontalization/Numpy/lsm_y.npy'
Y_test = np.load(Y_test_path.split('\u202a')[0])

X_test_list = [] #
Y_test_list = [] #

for i in range(n_test_image): #
    X_test_list.append(X_test) #
    Y_test_list.append(Y_test) #

X_test = np.array(X_test_list) #
Y_test = np.array(Y_test_list) #

# print(X_test.shape)
# print(Y_test.shape)

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
Y_train = Y_train / 127.5 - 1.
X_test = X_test / 127.5 - 1.
Y_test = Y_test / 127.5 - 1.

# Prameters
height = X_train.shape[1]
width = X_train.shape[2]
channels = X_train.shape[3]
latent_dimension = width

quarter_height = int(np.round(np.round(height / 2) / 2))
quarter_width = int(np.round(np.round(width / 2) / 2))
half_latent_dimension = int(round(latent_dimension / 2))

# print(height)
# print(width)
# print(quarter_height)
# print(quarter_width)
# print(channels)
# print(latent_dimension)

optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

n_show_image = 1 # Number of images to show

number = 0

train_epochs = 10000
test_epochs = 1
train_batch_size = latent_dimension
test_batch_size = latent_dimension
train_save_interval = 1
test_save_interval = 1

def generator_first_filter():
    if latent_dimension > 64:
        generator_first_filter = 64
        return generator_first_filter
    else:
        generator_first_filter = latent_dimension
        return generator_first_filter

print(generator_first_filter())