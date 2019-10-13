from __future__ import print_function, division

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Activation, add, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

np.random.seed(10)

time = 1

# Load data
X_train = np.load('D:/Bitcamp/Project/Frontalization/Imagenius/Numpy/korean_lux_x.npy') # Side face
Y_train = np.load('D:/Bitcamp/Project/Frontalization/Imagenius/Numpy/korean_lux_y.npy') # Front face

# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

train_epochs = 100
batch_size = 32
save_interval = 1

class DCGAN():
    def __init__(self):
        # Rescale -1 to 1
        self.X_train = X_train / 127.5 - 1.
        self.Y_train = Y_train / 127.5 - 1.
        # X_test = X_test / 127.5 - 1.
        # Y_test = Y_test / 127.5 - 1.

        # Prameters
        self.height = X_train.shape[1]
        self.width = X_train.shape[2]
        self.channels = X_train.shape[3]
        self.latent_dimension = self.width

        self.optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

        self.batch = int(self.X_train.shape[0] / batch_size)

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 1
        self.save_path = 'D:/Generated Image/Training' + str(time) + '/'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss = self.vgg19_loss, optimizer = self.optimizer)

        # Save .json
        generator_model_json = self.generator.to_json()

        # Check folder presence
        if not os.path.isdir(self.save_path + 'Json/'):
            os.makedirs(self.save_path + 'Json/')

        with open(self.save_path + 'Json/generator_model.json', "w") as json_file : 
            json_file.write(generator_model_json)

        # The generator takes side images as input and generates images
        z = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, [image, valid])
        self.combined.compile(loss = [self.vgg19_loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer = self.optimizer)

        # self.combined.summary()

    def residual_block(self, layer, filters, kernel_size, strides):
        generator = layer

        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(generator)
        layer = BatchNormalization(momentum = 0.5)(layer)

        # Using Parametric ReLU
        layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = 'same')(layer)
        output = BatchNormalization(momentum = 0.5)(layer)

        model = add([generator, output])

        return model

    def up_sampling_block(self, layer, filters, kernel_size, strides):
        # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
        # Even we can have our own function for deconvolution (i.e one made in Utils.py)
        # layer = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = 'same)(layer)
        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(layer)
        layer = UpSampling2D(size = (2, 2))(layer)
        layer = LeakyReLU(alpha = 0.2)(layer)

        return layer
    
    # computes VGG loss or content loss
    def vgg19_loss(self, true, prediction):
        vgg19 = VGG19(include_top = False, weights = 'imagenet', input_shape = (self.height, self.width, self.channels))
        # Make trainable as False

        vgg19.trainable = False

        for layer in vgg19.layers:
            layer.trainable = False
        
        model = Model(inputs = vgg19.input, outputs = vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(true) - model(prediction)))

    def build_generator(self):
        generator_input = Input(shape = (self.height, self.width, self.channels))

        generator_layer = Conv2D(filters = 16, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_input)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)
        generator_layer = Conv2D(filters = 32, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)
        generator_layer = Conv2D(filters = 64, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)

        previous_output = generator_layer

        # Using 16 Residual Blocks
        for i in range(16):
            generator_layer = self.residual_block(layer = generator_layer, filters = 64, kernel_size = (3, 3), strides = (1, 1))

        generator_layer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.5)(generator_layer)
        generator_layer = add([previous_output, generator_layer])

        # Using 2 UpSampling Blocks
        for j in range(3):
            generator_layer = self.up_sampling_block(layer = generator_layer, filters = 256, kernel_size = (3, 3), strides = (1, 1))

        generator_layer = Conv2D(filters = self.channels, kernel_size = (9, 9), strides = (1, 1), padding = 'same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        generator = Model(inputs = generator_input, outputs = generator_output)

        # generator.summary()

        return generator

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size = (3, 3), strides = (1, 1), input_shape = (self.height, self.width, self.channels), padding = 'same'))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(256, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(512, kernel_size = (3, 3), strides = (1, 1), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dense(1, activation = 'sigmoid'))

        # model.summary()

        generated_image = Input(shape = (self.height, self.width, self.channels))
        validity = model(generated_image)

        discriminator = Model(generated_image, validity)

        # discriminator.summary()

        return discriminator

    def train(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Training')

        for k in range(1, epochs + 1):
            for l in tqdm(range(1, self.batch + 1)):
                # Select a random half of images
                index = np.random.randint(0, self.X_train.shape[0], batch_size)
                front_image = self.Y_train[index]

                # Generate a batch of new images
                side_image = self.X_train[index]
               
                generated_image = self.generator.predict(side_image)

                self.discriminator.trainable = True

                # Train the discriminator (real classified as ones and generated as zeros)
                discriminator_fake_loss = self.discriminator.train_on_batch(generated_image, fake)
                discriminator_real_loss = self.discriminator.train_on_batch(front_image, real)
                discriminator_loss = 0.5 * np.add(discriminator_fake_loss, discriminator_real_loss)
                
                self.discriminator.trainable = False

                # Train the generator (wants discriminator to mistake images as real)
                generator_loss = self.combined.train_on_batch(side_image, [front_image, real])

                # Plot the progress
                print ('\nTraining epoch : %d \nTraining batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f ' 
                        % (k, l, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2]))

                record = (k, l, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2])
                self.history.append(record)

                # If at save interval -> save generated image samples
                if l % save_interval == 0:
                    self.save_image(front_image = front_image, side_image = side_image, train_number = k, epoch_number = l, save_path = self.save_path)

            # Save .h5
            if k % 5 == 0:
                # Check folder presence
                if not os.path.isdir(self.save_path + 'H5/'):
                    os.makedirs(self.save_path + 'H5/')

                self.generator.save(self.save_path + 'H5/' + 'generator_epoch_%d.h5' % k)
                self.generator.save_weights(self.save_path + 'H5/' + 'generator_weights_epoch_%d.h5' % k)

        self.history = np.array(self.history)

        self.graph(history = self.history, save_path = self.save_path + 'History/')

    def save_image(self, front_image, side_image, train_number, epoch_number, save_path):
        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
        for m in range(batch_size):
            plt.figure(figsize = (8, 2))

            # Adjust the interval of the image
            plt.subplots_adjust(wspace = 0.6)

            for n in range(self.n_show_image):
                generated_image_plot = plt.subplot(1, 3, n + 1 + (2 * self.n_show_image))
                generated_image_plot.set_title('Generated image (front image)')

                if self.channels == 1:
                    plt.imshow(generated_image[m,  :  ,  :  , 0], cmap = 'gray')
                
                else:
                    plt.imshow(generated_image[m,  :  ,  :  ,  : ])

                original_front_face_image_plot = plt.subplot(1, 3, n + 1 + self.n_show_image)
                original_front_face_image_plot.set_title('Origninal front image')

                if self.channels == 1:
                    plt.imshow(front_image[m].reshape(self.height, self.width), cmap = 'gray')
                    
                else:
                    plt.imshow(front_image[m])

                original_side_face_image_plot = plt.subplot(1, 3, n + 1)
                original_side_face_image_plot.set_title('Origninal side image')

                if self.channels == 1:
                    plt.imshow(side_image[m].reshape(self.height, self.width), cmap = 'gray')
                    
                else:
                    plt.imshow(side_image[m])

                # Don't show axis of x and y
                generated_image_plot.axis('off')
                original_front_face_image_plot.axis('off')
                original_side_face_image_plot.axis('off')

                self.number += 1

                # plt.show()

            save_path = save_path

            # Check folder presence
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_name = 'Train%d_Batch%d_%d.png' % (train_number, epoch_number, self.number)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()

    def graph(self, history, save_path):
        plt.plot(self.history[  :  , 2])     
        plt.plot(self.history[  :  , 3])
        plt.plot(self.history[  :  , 4])
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

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs = train_epochs, batch_size = batch_size, save_interval = save_interval)