from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization,Conv2D, Dense, Dropout, Flatten, Input, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load data
X = np.load('./npy/x.npy') # Side face
Y = np.load('./npy/y.npy') # Front face

# print(X.shape) # (5400, 28, 28, 1)
# print(Y.shape) # (5400, 28, 28, 1)

# Prameters
height = X.shape[1]
width = X.shape[2]
channels = X.shape[3]
latent_dimension = 28

optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

class DCGAN():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.latent_dimension = latent_dimension

        self.optimizer = optimizer

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape = (self.latent_dim, ))
        image = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512 * 8 * 8, activation = LeakyReLU, input_dim = self.latent_dimension))
        model.add(Reshape(8, 8, 512))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(Activation(LeakyReLU))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(Activation(LeakyReLU))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(Activation(LeakyReLU))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channel, kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('tanh'))

        model.summary()
        
        noise = Input(shape = self.latent_dimension, )
        image = model(noise)
        
        return Model(noise, image)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size = 3, strides = 1, input_shape = (self.height, self.width, self.channel), padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(ZeroPadding2D(padding=((0, 1),(0, 1))))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        model.summary()

        image = Input(shape = (self.height, self.width, self.channel))
        validity = model(image)

        return Model(image, validity)

    def train(self, epochs, batch_size = 128, save_interval = 50):
        # Rescale -1 to 1
        X_train = X / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        Y = Y.reshae(Y.shape[0], Y.shape[1], Y.shape[2])