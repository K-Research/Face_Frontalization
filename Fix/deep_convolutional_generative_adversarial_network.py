from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization,Conv2D, Dense, Dropout, Flatten, Input, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

n_test_image = 2
time = 1

# Load data
X_train = np.load('D:/Bitcamp/BitProject/npy/x.npy')
Y_train = np.load('D:/Bitcamp/BitProject/npy/y.npy')

# print(X_train.shape) # (300, 28, 28, 1)
# print(Y_train.shape) # (300, 28, 28, 1)

X_test = np.load('D:/Bitcamp/BitProject/npy/lsm_x.npy') # Side face
# Y_test = np.load('‪D:/Bitcamp/BitProject/npy/lsm_y.npy') # Front face
Y_test_path = '‪D:/Bitcamp/BitProject/npy/lsm_y.npy'
Y_test = np.load(Y_test_path.split('\u202a')[1])

X_test_list = [] #
Y_test_list = [] #

for i in range(n_test_image): #
    X_test_list.append(X_test) #
    Y_test_list.append(Y_test) #

X_test = np.array(X_test_list) #
Y_test = np.array(Y_test_list) #

# print(X_test.shape)
# print(Y_test.shape)

# Prameters
height = X_train.shape[1]
width = X_train.shape[2]
channels = X_train.shape[3]

latent_dimension = width

# print(height)
# print(width)
# print(channels)
# print(latent_dimension)

optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

n_show_image = 1 # Number of images to show

image_index = 0

train_epochs = X_train.shape[0]
test_epochs = X_test.shape[0]
train_save_interval = 1
test_save_interval = 1

X_train = X_train.reshape(train_epochs, height, width)
X_test = X_test.reshape(test_epochs, height, width)

def paramertic_relu(alpha_initializer, alpha_regularizer, alpha_constraint, shared_axes):
    PReLU(alpha_initializer = alpha_initializer, alpha_regularizer = alpha_regularizer, alpha_constraint = alpha_constraint, shared_axes = shared_axes)

class DCGAN():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        # self.image_shape = (self.height, self.width, self.channels)
        self.latent_dimension = latent_dimension

        self.optimizer = optimizer

        self.image_index = image_index

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape = (self.latent_dimension, ))
        # z = Input(shape = (self.height, self.width, self.channels, )) #
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
        model.add(Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channels, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(Activation('tanh'))

        # model.summary()
        
        side_face = Input(shape = (self.latent_dimension, ))
        image = model(side_face)
        
        return Model(side_face, image)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size = (3, 3), strides = (2, 2), input_shape = (self.height, self.width, self.channels), padding = 'same'))
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
        model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))

        model.summary()

        image = Input(shape = (self.height, self.width, self.channels))
        validity = model(image)

        return Model(image, validity)

    def train(self, epochs, batch_size, save_interval):
        # Rescale -1 to 1
        y_train = Y_train / 127.5 - 1.

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Training')

        for i in range(epochs):
            # Select a random half of images
            index = np.random.randint(0, train_epochs, batch_size)
            front_image = y_train[index]

            # Generate a batch of new images
            side_image = X_train[i]
                        
            generated_image = self.generator.predict(side_image)

            self.discriminator.trainable = True

            # Train the discriminator (real classified as ones and generated as zeros)
            discriminator_fake_loss = self.discriminator.train_on_batch(generated_image, fake)
            discriminator_real_loss = self.discriminator.train_on_batch(front_image, real)
            discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)

            self.discriminator.trainable = False

            # Train the generator (wants discriminator to mistake images as real)
            generator_loss = self.combined.train_on_batch(side_image, real)
            
            # Plot the progress
            print ('Training epoch : %d  \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f'
                    % (i, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss))

            # If at save interval -> save generated image samples
            if i % save_interval == 0:
                save_path = 'D:/Generated Image/Training' + str(time) + '/'
                self.save_image(number = i, front_image = front_image, side_image = side_image, save_path = save_path)

    def test(self, epochs, batch_size, save_interval):
        # Rescale -1 to 1
        y_test = Y_test / 127.5 - 1.

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Testing')

        for j in range(epochs):
            # Select a random half of images
            index = np.random.randint(0, test_epochs, batch_size)
            front_image = y_test[index]

            # Generate a batch of new images
            side_image = X_train[index] 

            generated_image = self.generator.predict(side_image)

            # Train the discriminator (real classified as ones and generated as zeros)
            discriminator_fake_loss = self.discriminator.test_on_batch(generated_image, fake)
            discriminator_real_loss = self.discriminator.test_on_batch(front_image, real)
            discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)

            # Train the generator (wants discriminator to mistake images as real)
            generator_loss = self.combined.test_on_batch(side_image, real)
            
            # Plot the progress
            print ('Test epoch : %d  \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f'
                    % (j, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss))

            # If at save interval -> save generated image samples
            if i % save_interval == 0:
                save_path = 'D:/Generated Image/Testing' + str(time) + '/'
                self.save_image(number = j, front_image = front_image, side_image = side_image, save_path = save_path)

    def save_image(self, number, front_image, side_image, save_path):
        # global image_index

        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        plt.figure(figsize = (8, 2))

        # Adjust the interval of the image
        plt.subplots_adjust(wspace = 0.6)

        # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
        for k in range(n_show_image):
            generated_image_plot = plt.subplot(1, 3, k + 1 + (2 * n_show_image))
            generated_image_plot.set_title('Generated image (front image)')

            if channels == 1:
                plt.imshow(generated_image[image_index,  :  ,  :  , 0], cmap = 'gray')
            
            else:
                plt.imshow(generated_image[image_index,  :  ,  :  ,  : ])

            original_front_face_image_plot = plt.subplot(1, 3, k + 1 + n_show_image)
            original_front_face_image_plot.set_title('Origninal front image')

            if channels == 1:
                plt.imshow(front_image[image_index].reshape(height, width), cmap = 'gray')
                
            else:
                plt.imshow(front_image[image_index], cmap = 'gray')

            original_side_face_image_plot = plt.subplot(1, 3, k + 1)
            original_side_face_image_plot.set_title('Origninal side image')

            if channels == 1:
                # plt.imshow(side_image[image_index].reshape(height, width), cmap = 'gray')
                plt.imshow(side_image, cmap = 'gray')
                
            else:
                plt.imshow(side_image[image_index], cmap = 'gray')

            # Don't show axis of x and y
            generated_image_plot.axis('off')
            original_front_face_image_plot.axis('off')
            original_side_face_image_plot.axis('off')

            # plt.show()

        save_path = save_path

        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_name = '%d.png' % number
        save_name = os.path.join(save_path, save_name)
        plt.savefig(save_name)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs = train_epochs, batch_size = 28, save_interval = 1)
    dcgan.test(epochs = test_epochs, batch_size = 28, save_interval = 1)