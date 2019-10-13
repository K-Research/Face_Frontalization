from __future__ import print_function, division

from datagenerator_read_dir_face import DataGenerator
from glob import glob
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Activation, add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

senet50_layer = VGGFace(include_top = False, model = 'senet50', weights = 'vggface', input_shape = (224, 224, 3))

senet50_layer.summary()

senet50_last_layer = senet50_layer.get_layer('activation_81').output

discriminator_layer = Conv2D(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(senet50_last_layer)
discriminator_layer = Conv2D(filters = 256, kernel_size = (2, 2), strides = (1, 1), padding = 'valid')(discriminator_layer)

discriminator = Model(inputs = senet50_layer.input, outputs = discriminator_layer)

discriminator.summary()