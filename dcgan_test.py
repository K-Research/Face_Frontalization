from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization,Conv2D, Dense, Dropout, Flatten, Input, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
import sys

time = 1

# Load data
X = np.load('D:/Taehwan Kim/Document/Bitcamp/BitProject/npy/x.npy') # Side face
Y = np.load('D:/Taehwan Kim/Document/Bitcamp/BitProject/npy/y.npy') # Front face


# print(X.shape) # (5400, 28, 28, 1)
# print(Y.shape) # (5400, 28, 28, 1)

X_train = X.reshape(X.shape[0], X.shape[1], X.shape[2])

X_test = np.load('D:/Taehwan Kim/Document/Bitcamp/BitProject/npy/lsm_x.npy')
# Y_test = np.load('‪D:/Taehwan Kim/Document/Bitcamp/BitProject/npy/lsm_y.npy')
Y_test_path = '‪D:/Taehwan Kim/Document/Bitcamp/BitProject/npy/lsm_y.npy'
Y_test = np.load(Y_test_path.split("\u202a")[1])

X_test_list = [] #
Y_test_list = [] #

for i in range(28): #
    X_test_list.append(X_test) #
    Y_test_list.append(Y_test) #

X_test = np.array(X_test_list) #
Y_test = np.array(Y_test_list) #

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# print(X_test.shape) # (28, 28, 28)
# print(Y_test.shape) # (28, 28, 28, 1)

# Shuffle
X, Y = shuffle(X, Y, random_state = 66)

# Prameters
height = X.shape[1]
width = X.shape[2]
channels = X.shape[3]

def latent_dimension():
    if height == width:
        latent_dimension = height
    else:
        latent_dimension = int(math.sqrt(height * width))


# print(height) # 28
# print(width) # 28
# print(channels) # 1
# print(latent_dimension) # 28

optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
train_epochs = X.shape[0]
test_epochs = X_test.shape[0]

def paramertic_relu(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes):
    PReLU(alpha_initializer = alpha_initializer, alpha_regularizer = alpha_regularizer, alpha_constraint = alpha_constraint, shared_axes = shared_axes)

class DCGAN():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        # self.image_shape = (self.height, self.width, self.channels)
        self.latent_dimension = latent_dimension()

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
        # model.add(Dense(128 * 7 * 7, activation = paramertic_relu(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = None), input_dim = self.latent_dimension))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(Activation('relu'))
        # model.add(Activation(paramertic_relu(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.5))
        model.add(Activation('relu'))
        # model.add(Activation(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])))
        model.add(Conv2D(self.channels, kernel_size = (3, 3), strides = 1, padding = 'same'))
        # model.add(Conv2D(self.channels, kernel_size = (9, 9), strides = 1, padding = 'same'))
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
        # model.add(Conv2D(64, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(ZeroPadding2D(padding=((0, 1),(0, 1))))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size = (3, 3), strides = 2, padding = 'same'))
        # model.add(Conv2D(128, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size = (3, 3), strides = 2, padding = 'same'))
        # model.add(Conv2D(256, kernel_size = (3, 3), strides = 1, padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        # model.summary()

        image = Input(shape = (self.height, self.width, self.channels))
        validity = model(image)

        return Model(image, validity)

    def train(self, epochs, batch_size, save_interval):
        # Rescale -1 to 1
        Y_train = Y / 127.5 - 1.
        # Y_train = np.expand_dims(Y_train, axis = 3)

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Training')

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

            # Train the generator (wants discriminator to mistake images as real)
            generator_loss = self.combined.train_on_batch(x_train, real)
            
            # Plot the progress
            print ('Training epoch : %d  \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f' % (i, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss))

            # If at save interval -> save generated image samples
            if i % save_interval == 0:
                save_path = 'D:/Training' + str(time) + '/'
                self.save_image(i, save_path)

    def test(self, epochs, batch_size, save_interval):
        # Rescale -1 to 1
        y_test = Y_test / 127.5 - 1.
        # Y_test = np.expand_dims(Y_test, axis = 3)

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Testing')

        for i in range(epochs):
            # Select a random half of images
            index = np.random.randint(0, y_test.shape[0], batch_size)
            front_image = y_test[index]

            # Sample noise and generate a batch of new images
            x_test = X_test[i]

            generated_image = self.generator.predict(x_test)

            # Train the discriminator (real classified as ones and generated as zeros)
            discriminator_fake_loss = self.discriminator.test_on_batch(generated_image, fake)
            discriminator_real_loss = self.discriminator.test_on_batch(front_image, real)
            discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)

            # Train the generator (wants discriminator to mistake images as real)
            generator_loss = self.combined.test_on_batch(x_test, real)
            
            # Plot the progress
            print ('Test epoch : %d  \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f' % (i, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss))

            # If at save interval -> save generated image samples
            if i % save_interval == 0:
                save_path = 'D:/Testing' + str(time) + '/'
                self.save_image(i, save_path)
      
    def save_image(self, number, save_path):
        row, column = 5, 5

        # generated_image = self.generator.predict(X_train[number])

        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(X_train[number]) + 0.5
        
        figure, axis = plt.subplots(1, 2)

        count = 0

        for j in range(row):
            for k in range(column):
                axis[j, k].imshow(generated_image[count, :  , :  , 0], cmap = 'gray')
                axis[j, k].axis('off')
                count += 1
                
        # plt.show()

        save_path = save_path

        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_image = '%d.png' % number
        save_image = os.path.join(save_path, save_image)
        figure.savefig(save_image)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    # dcgan.train(epochs = train_epochs, batch_size = 28, save_interval = 1)
    dcgan.train(epochs = 1, batch_size = 28, save_interval = 1)
    # dcgan.test(epochs = test_epochs, batch_size = 28, save_interval = 1)
    dcgan.test(epochs = 1, batch_size = 28, save_interval = 1)