from __future__ import print_function, division

from datagenerator_read_dir_face import DataGenerator
from glob import glob
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D, Dropout
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

# senet50_layer = VGGFace(include_top = False, model = 'senet50', weights = 'vggface', input_shape = (224, 224, 3))

# senet50_last_layer = senet50_layer.get_layer('avg_pool').output
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(senet50_last_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
# generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)

# generator_output = Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = (1, 1), padding = 'valid')(generator_layer)

# generator = Model(inputs = senet50_layer.input, outputs = generator_output)

# generator.summary()

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), strides = (2, 2), input_shape = (224, 224, 3), padding = 'same'))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(ZeroPadding2D(padding = ((0, 1), (0, 1))))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
model.add(BatchNormalization(momentum = 0.8))
model.add(LeakyReLU(alpha = 0.2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()