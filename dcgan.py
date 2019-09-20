from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization,Conv2D, Dense, Dropout, Flatten, Input, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

test_times = 1

# Load data
X = np.load('D:/Bitcamp/BitProject/npy/x.npy') # Side face
Y = np.load('D:/Bitcamp/BitProject/npy/y.npy') # Front face

# print(X.shape) # (5400, 28, 28, 1)
# print(Y.shape) # (5400, 28, 28, 1)

X_train = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Prameters
height = X.shape[1]
width = X.shape[2]
channels = X.shape[3]
latent_dimension = int(math.sqrt(height * width))

# print(height) # 28
# print(width) # 28
# print(channels) # 1
# print(latent_dimension) # 28

optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
'''
# Shuffle
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state = 66)
'''
class DCGAN():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        # self.image_shape = (self.height, self.width, self.channels)
        self.latent_dimension = latent_dimension

        self.optimizer = optimizer

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape = (self.latent_dimension, ))
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

        model.add(Dense(128 * 7 * 7, activation = 'relu', input_dim = self.latent_dimension))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels, kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('tanh'))

        # model.summary()
        
        side_face = Input(shape = (self.latent_dimension, ))
        image = model(side_face)
        
        return Model(side_face, image)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size = 3, strides = 2, input_shape = (self.height, self.width, self.channels), padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size = (3, 3), strides = 2, padding = 'same'))
        model.add(ZeroPadding2D(padding=((0, 1),(0, 1))))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size = (3, 3), strides = 2, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size = (3, 3), strides = 2, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        # model.summary()

        image = Input(shape = (self.height, self.width, self.channels))
        validity = model(image)

        return Model(image, validity)

    def train(self, epochs, batch_size = 128, save_interval = 50):
        # Rescale -1 to 1
        Y_train = Y / 127.5 - 1.
        # Y_train = np.expand_dims(Y_train, axis=3)

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        for i in range(epochs):
            # Select a random half of images
            index = np.random.randint(0, Y_train.shape[0], batch_size)
            front_image = Y_train[index]

            # Sample noise and generate a batch of new images
            x_train = X_train[i]

            generated_image = self.generator.predict(x_train)

            # Train the discriminator (real classified as ones and generated as zeros)
            discriminator_fake_loss = self.discriminator.train_on_batch(generated_image, fake)
            discriminator_real_loss = self.discriminator.train_on_batch(front_image, real)
            discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)
            # discriminator_real_loss = self.discriminator.train_on_batch(image, real)
            # discriminator_fake_loss = self.discriminator.train_on_batch(generated_image, fake)
            # discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)

            # Train the generator (wants discriminator to mistake images as real)
            generator_loss = self.combined.train_on_batch(x_train, real)
            
            # Plot the progress
            print ("%d [Loss of discriminator : %f, Accuracy : %.2f%%] [Loss of generator : %f]" % (i, discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))

            # If at save interval -> save generated image samples
            if i % save_interval == 0:
                self.save_image(i)
      
    def save_image(self, number):
        row, column = 5, 5

        # generated_image = self.generator.predict(X_train[number])

        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(X_train[number]) + 0.5

        figure, axis = plt.subplots(row, column)

        count = 0

        for j in range(row):
            for k in range(column):
                axis[j, k].imshow(generated_image[count, :  , :  , 0], cmap = 'gray')
                axis[j, k].axis('off')
                count += 1

        save_path = 'D:/Test' + str(test_times) + '/'

        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_filename = '%d.png' % number
        save_filename = os.path.join(save_path, save_filename)
        figure.savefig(save_filename)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs = 5400, batch_size = 28, save_interval = 1)