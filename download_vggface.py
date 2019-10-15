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

generator_input = Input(shape = (8631, )) # 8631

# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 2), strides = (1, 1), padding = 'valid')(generator_input)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 2), strides = (1, 1), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 8, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
# generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
# generator_layer = Conv2DTranspose(filters = 3, kernel_size = (3, 3), strides = (1, 1), padding = 'valid')(generator_layer)

generator_output = Activation('tanh')(generator_input)

generator = Model(inputs = generator_input, outputs = generator_output)

generator.summary()