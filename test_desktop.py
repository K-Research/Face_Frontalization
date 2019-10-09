from __future__ import print_function, division

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.engine.topology import Network
from keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, InputSpec, Layer, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
import tensorflow as tf
from tqdm import tqdm

np.random.seed(seed = 12345)

time = 1

# Load data
X_train = np.load('D:/Taehwan Kim/Document/Bitcamp/Project/Frontalization/Imagenius/Numpy/korean_lux_x.npy') # Side face
Y_train = np.load('D:/Taehwan Kim/Document/Bitcamp/Project/Frontalization/Imagenius/Numpy/korean_lux_y.npy') # Front face

# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

# Shuffle
# X_train, Y_train = shuffle(X_train, Y_train, random_state = 66)
# X_test, Y_test = shuffle(X_test, Y_test, random_state = 66)

train_epochs = 10000
test_epochs = 1
train_batch_size = 16
test_batch_size = 1
train_save_interval = 1
test_save_interval = 1

class DCGAN():
    def __init__(self):
        # Rescale -1 to 1
        self.X_train = X_train / 127.5 - 1.
        self.Y_train = Y_train / 127.5 - 1.
        # X_test = X_test / 127.5 - 1.
        # Y_test = Y_test / 127.5 - 1.

        # Prameters
        self.height = self.X_train.shape[1]
        self.width = self.X_train.shape[2]
        self.channels = self.X_train.shape[3]
        self.latent_dimension = self.width

        self.discriminator_optimizer = Adam(lr = 2e-4, beta_1 = 0.5, beta_2 = 0.999)
        self.generator_optimizer = Adam(lr = 2e-4, beta_1 = 0.5, beta_2 = 0.999)

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 0

        # Build and compile the discriminator
        discriminator_A = self.build_discriminator()
        discriminator_B = self.build_discriminator()

        image_A = Input(shape = (self.height, self.width, self.channels))
        image_B = Input(shape = (self.height, self.width, self.channels))

        guess_A = discriminator_A(image_A)
        guess_B = discriminator_B(image_B)

        self.discriminator_A = Model(inputs = image_A, outputs = guess_A)
        self.discriminator_B = Model(inputs = image_B, outputs = guess_B)

        self.discriminator_A.compile(optimizer = self.discriminator_optimizer, loss = self.least_squares_error, loss_weights = [0.5])
        self.discriminator_B.compile(optimizer = self.discriminator_optimizer, loss = self.least_squares_error, loss_weights = [0.5])

        # Use Networks to avoid falsy keras error about weight descripancies
        self.discriminator_A_static = Network(inputs = image_A, outputs = guess_A)
        self.discriminator_B_static = Network(inputs = image_B, outputs = guess_B)

        # Do note update discriminator weights during generator training
        self.discriminator_A_static.trainable = False
        self.discriminator_B_static.trainable = False

        # Build and compile the generator
        self.generator_A_to_B = self.build_generator()
        self.generator_B_to_A = self.build_generator()

        # If use identity learning
        # self.generator_A_to_B.compile(optimizer = self.generator_optimizer, loss = 'mae')
        # self.generator_B_to_A.compile(optimizer = self.generator_optimzier, loss = 'mae')

        # Generator builds
        real_A = Input(shape = (self.height, self.width, self.channels))
        real_B = Input(shape = (self.height, self.width, self.channels))

        synthetic_A = self.generator_B_to_A(real_B)
        synthetic_B = self.generator_A_to_B(real_A)

        discriminator_A_guess_synthetic = self.discriminator_A_static(synthetic_A)
        discriminator_B_guess_synthetic = self.discriminator_B_static(synthetic_B)

        reconstructed_A = self.generator_B_to_A(synthetic_B)
        reconstructed_B = self.generator_A_to_B(synthetic_A)

        model_output = [reconstructed_A, reconstructed_B]
        compile_loss = [self.cycle_loss, self.cycle_loss, self.least_squares_error, self.least_squares_error]
        compile_weights = [10.0, 10.0, 1.0, 1.0]

        model_output.append(discriminator_A_guess_synthetic)
        model_output.append(discriminator_B_guess_synthetic)

        self.generator = Model(inputs = [real_A, real_B], outputs = model_output)
        self.generator.compile(optimizer = self.generator_optimizer, loss = compile_loss, loss_weights = compile_weights)

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))

        return loss

    def least_squares_error(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        
        return loss

    def residual_block(self, previous_layer):
        filters = int(previous_layer.shape[-1])

        # First layer
        layer = ReflectionPadding2D((1, 1))(previous_layer)
        layer = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'valid')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)

        # Second layer
        layer = ReflectionPadding2D((1, 1))(layer)
        layer = Conv2D(filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'valid')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer) 

        # Merge
        layer = add([layer, previous_layer])

        return layer

    def build_generator(self):
        # Specify input
        input = Input(shape = (self.height, self.width, self.channels))

        layer = ReflectionPadding2D((3, 3))(input)
        layer = Conv2D(filters = 32, kernel_size = (7, 7), strides = (1, 1), padding = 'valid')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)

        # Residual layer
        for _ in range(9):
            layer = self.residual_block(layer)

        layer = Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)
        layer = Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        # layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Activation('relu')(layer)
        layer = ReflectionPadding2D((3, 3))(layer)
        layer = Conv2D(filters = self.channels, kernel_size = (7, 7), strides = (1, 1))(layer)

        output = Activation('tanh')(layer)

        model = Model(inputs = input, outputs = output)

        # model.summary()

        return model

    def build_discriminator(self):
        # Specify input
        input = Input(shape = (self.height, self.width, self.channels))

        layer = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(input)
        layer = LeakyReLU(alpha = 0.2)(layer)
        layer = Conv2D(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        layer = LeakyReLU(alpha = 0.2)(layer)
        layer = Conv2D(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        layer = LeakyReLU(alpha = 0.2)(layer)
        layer = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer)
        layer = InstanceNormalization(axis = 3, epsilon = 1e-5, center = True)(layer, training = True)
        layer = LeakyReLU(alpha = 0.2)(layer)
        layer = Flatten()(layer)
        
        output = Dense(units = 1, activation = 'sigmoid')(layer)

        model = Model(inputs = input, outputs = output)

        # model.summary()

        return model

    def train(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        synthetic_pool_A = ImagePool(50)
        synthetic_pool_B = ImagePool(50)

        print('Training')

        for k in range(epochs):
            for l in tqdm(range(batch_size)):
                # Select a random half of images
                index = np.random.randint(0, self.X_train.shape[0], batch_size)
                front_image = self.Y_train[index]

                # Generate a batch of new images
                side_image = self.X_train[index]

                # target_data = [side_image, front_image]
                target_data = [front_image, side_image]


                synthetic_images_A = self.generator_B_to_A.predict(front_image)
                synthetic_images_B = self.generator_A_to_B.predict(side_image)
                synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
                synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

                # Train the discriminator (real classified as ones and generated as zeros)
                discriminator_A_loss_real = self.discriminator_A.train_on_batch(x = side_image, y = real)
                discriminator_B_loss_real = self.discriminator_B.train_on_batch(x = front_image, y = real)

                discriminator_A_loss_synthetic = self.discriminator_A.train_on_batch(x = synthetic_images_A, y = fake)
                discriminator_B_loss_synthetic = self.discriminator_B.train_on_batch(x = synthetic_images_B, y = fake)

                discriminator_A_loss = discriminator_A_loss_real + discriminator_A_loss_synthetic
                discriminator_B_loss = discriminator_B_loss_real + discriminator_B_loss_synthetic

                target_data.append(real)
                target_data.append(real)

                # Train the generator (wants discriminator to mistake images as real)
                generator_loss = self.generator.train_on_batch(x = [side_image, front_image], y = target_data)

                generator_A_discriminator_loss_synthetic = generator_loss[1]
                generator_B_discriminator_loss_synthetic = generator_loss[2]

                reconstruction_A_loss = generator_loss[3]
                reconstruction_B_loss = generator_loss[4]


                # Plot the progress
                print ('\nTraining epoch : %d \nTraining batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f ' 
                        % (k + 1, l + 1, discriminator_B_loss * 100, discriminator_B_loss, generator_B_discriminator_loss_synthetic))

                record = (k + 1, l + 1, discriminator_B_loss * 100, discriminator_B_loss, generator_B_discriminator_loss_synthetic)
                self.history.append(record)

                # If at save interval -> save generated image samples
                if l % save_interval == 0:
                    save_path = 'D:/Generated Image/Training' + str(time) + '/'
                    self.save_image(image_index = l, front_image = front_image, side_image = side_image, save_path = save_path)

        self.history = np.array(self.history)

        self.graph(history = history, save_path = save_path)

    def test(self, epochs, batch_size, save_interval):
        global history

        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Testing')

        for m in range(epochs):
            for n in tqdm(range(batch_size)):
                # Select a random half of images
                index = np.random.randint(0, X_test.shape[0], batch_size)
                front_image = Y_test[index]

                # Generate a batch of new images
                side_image = X_test[index]

                generated_image = self.generator.predict(side_image)

                # Train the discriminator (real classified as ones and generated as zeros)
                discriminator_fake_loss = self.discriminator.test_on_batch(generated_image, fake)
                discriminator_real_loss = self.discriminator.test_on_batch(front_image, real)
                discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)

                # Train the generator (wants discriminator to mistake images as real)
                generator_loss = self.combined.test_on_batch(side_image, [front_image, real])
                
                # Plot the progress
                print ('\nTest epoch : %d \nTest batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f ' 
                        % (m + 1, n + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2]))

                record = (m + 1, n + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2])
                history.append(record)

                # If at save interval -> save generated image samples
                if n % save_interval == 0:
                    save_path = 'D:/Generated Image/Testing' + str(time) + '/'
                    self.save_image(image_index = n, front_image = front_image, side_image = side_image, save_path = save_path)

        history = np.array(history)

        self.history(history = history, save_path = save_path)

    def save_image(self, image_index, front_image, side_image, save_path):
        # Rescale images 0 - 1
        generated_image_A_to_B = 0.5 * self.generator_A_to_B.predict(side_image) + 0.5
        generated_image_B_to_A = 0.5 * self.generator_B_to_A.predict(front_image) + 0.5

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        plt.figure(figsize = (8, 2))

        # Adjust the interval of the image
        plt.subplots_adjust(wspace = 0.6)

        # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
        for m in range(self.n_show_image):
            generated_image_A_to_B_plot = plt.subplot(1, 4, m + 1 + (2 * self.n_show_image))
            generated_image_A_to_B_plot.set_title('Generated image (front image)')

            if self.channels == 1:
                plt.imshow(generated_image_A_to_B[image_index,  :  ,  :  , 0], cmap = 'gray')
            
            else:
                plt.imshow(generated_image_A_to_B[image_index,  :  ,  :  ,  : ])

            generated_image_B_to_A_plot = plt.subplot(1, 4, m + 1 + (3 * self.n_show_image))
            generated_image_B_to_A_plot.set_title('Generated image (front image)')

            if self.channels == 1:
                plt.imshow(generated_image_B_to_A[image_index,  :  ,  :  , 0], cmap = 'gray')
            
            else:
                plt.imshow(generated_image_B_to_A[image_index,  :  ,  :  ,  : ])

            original_front_face_image_plot = plt.subplot(1, 4, m + 1 + self.n_show_image)
            original_front_face_image_plot.set_title('Origninal front image')

            if self.channels == 1:
                plt.imshow(front_image[image_index].reshape(self.height, self.width), cmap = 'gray')
                
            else:
                plt.imshow(front_image[image_index])

            original_side_face_image_plot = plt.subplot(1, 4, m + 1)
            original_side_face_image_plot.set_title('Origninal side image')

            if self.channels == 1:
                plt.imshow(side_image[image_index].reshape(self.height, self.width), cmap = 'gray')
                
            else:
                plt.imshow(side_image[image_index])

            # Don't show axis of x and y
            generated_image_A_to_B_plot.axis('off')
            generated_image_B_to_A_plot.axis('off')
            original_front_face_image_plot.axis('off')
            original_side_face_image_plot.axis('off')

            self.number += 1

            # plt.show()

        save_path = save_path

        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        save_name = '%d.png' % self.number
        save_name = os.path.join(save_path, save_name)
    
        plt.savefig(save_name)
        plt.close()

    def graph(self, history, save_path):
        plt.plot(self.history[:, 2])     
        plt.plot(self.history[:, 3])
        plt.plot(self.history[:, 4])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generative adversarial network')
        plt.legend(['Accuracy of discriminator', 'Loss of discriminator', 'Loss of generator'], loc = 'upper left')

        figure = plt.gcf()

        # plt.show()

        save_path = save_path

        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # save_name = '%d.png' % number
        save_name = 'History.png'
        save_name = os.path.join(save_path, save_name)

        figure.savefig(save_name)
        plt.close()

# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs = train_epochs, batch_size = train_batch_size, save_interval = train_save_interval)
    # dcgan.test(epochs = test_epochs, batch_size = test_batch_size, save_interval = test_save_interval)