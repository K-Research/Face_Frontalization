from __future__ import print_function, division

from datagenerator_read_dir_face import DataGenerator
from glob import glob
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

time = 21

# Load data
X_train = glob('D:/Bitcamp/Project/Frontalization/Imagenius/Data/Korean 224X224X3 filtering/X/*jpg')
Y_train = glob('D:/Bitcamp/Project/Frontalization/Imagenius/Data/Korean 224X224X3 filtering/Y/*jpg')

epochs = 100
batch_size = 16
save_interval = 1

class GAN():
    def __init__(self):
        # Load data
        self.datagenerator = DataGenerator(X_train, Y_train, batch_size = batch_size)

        # Prameters
        self.height = 224
        self.width = 224
        self.channels = 3

        self.optimizer = Adam(lr = 1e-4, beta_1 = 0.9, beta_2 = 0.999)

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 1
        self.save_path = 'D:/Generated Image/Training' + str(time) + '/'

        self.resolution_discriminator = self.build_resolution_discriminator()
        self.resolution_discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])

        # Build and compile the generator
        self.frontalization_generator = load_model('D:/Generated Image/Training20/H5/generator_epoch_80.h5')

        # Build and compile the resolution
        self.resolution_generator = self.build_resolution_generator()
        self.resolution_generator.compile(loss = self.vgg19_loss, optimizer = self.optimizer)

        # Save .json
        resolution_generator_model_json = self.resolution_generator.to_json()

        # Check folder presence
        if not os.path.isdir(self.save_path + 'Json/'):
            os.makedirs(self.save_path + 'Json/')

        with open(self.save_path + 'Json/resolution_generator_model.json', "w") as json_file :
            json_file.write(resolution_generator_model_json)

        # The generator takes noise as input and generates imgs
        z = Input(shape = (self.height, self.width, self.channels))
        resolution_image = self.resolution_generator(z)

        # For the combined model we will only train the generator
        self.resolution_discriminator.trainable = False

        # The discriminator takes generated images as input and determines validiy
        resolution_valid = self.resolution_discriminator(resolution_image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

        self.resolution_gan = Model(z, [resolution_image, resolution_valid])
        self.resolution_gan.compile(loss = [self.vgg19_loss, 'binary_crossentropy'], loss_weights = [1., 1e-3], optimizer = self.optimizer)

        # self.resolution_gan.summary()

    def residual_block(self, layer, filters, kernel_size, strides):
        residual_input = layer

        residual_layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(residual_input)
        residual_layer = BatchNormalization(momentum = 0.5)(residual_layer)

        # Using Parametric ReLU
        residual_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(residual_layer)
        residual_layer = Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = 'same')(residual_layer)
        residual_output = BatchNormalization(momentum = 0.5)(residual_layer)

        residual_model = add([residual_input, residual_output])

        return residual_model

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

    def build_resolution_generator(self):
        generator_input = Input(shape = (self.height, self.width, self.channels))

        generator_layer = Conv2D(filters = 32, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_input)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)
        generator_layer = Conv2D(filters = 64, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)
        generator_layer = Conv2D(filters = 128, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(generator_layer)
        generator_layer = MaxPooling2D(pool_size = (2, 2))(generator_layer)

        for k in range(16):
            residual_layer = self.residual_block(layer = generator_layer, filters = 128, kernel_size = (3, 3), strides = (1, 1))

        generator_layer = add([generator_layer, residual_layer])

        generator_layer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = UpSampling2D(size = (2, 2))(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = UpSampling2D(size = (2, 2))(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(generator_layer)
        generator_layer = UpSampling2D(size = (2, 2))(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2D(filters = self.channels, kernel_size = (9, 9), strides = (1, 1), padding = 'same')(generator_layer)

        generator_output = Activation('tanh')(generator_layer)

        generator = Model(inputs = generator_input, outputs = generator_output)

        # generator.summary()

        return generator

    def build_resolution_discriminator(self):
        discriminator_input = Input(shape = (self.height, self.width, self.channels))

        discriminator_layer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(discriminator_input)
        discriminator_layer = BatchNormalization(momentum = 0.5)(discriminator_layer)
        discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(discriminator_layer)
        discriminator_layer = BatchNormalization(momentum = 0.5)(discriminator_layer)
        discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(discriminator_input)
        discriminator_layer = BatchNormalization(momentum = 0.5)(discriminator_layer)
        discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(discriminator_layer)
        discriminator_layer = BatchNormalization(momentum = 0.5)(discriminator_layer)
        discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(discriminator_input)
        discriminator_layer = BatchNormalization(momentum = 0.5)(discriminator_layer)
        discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Flatten()(discriminator_layer)
        # discriminator_layer = Dense(units = 1024)(discriminator_layer)
        # discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        discriminator_layer = Dense(units = 1)(discriminator_layer)

        discriminator_output = Activation('sigmoid')(discriminator_layer)

        discriminator = Model(inputs = discriminator_input, outputs = discriminator_output)

        # discriminator.summary()

        return discriminator        

    def train(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        # fake = np.zeros((batch_size, 1))
        # real = np.ones((batch_size, 1))
        fake = np.random.random_sample((batch_size, 1)) * 0.2
        real = np.ones((batch_size, 1)) - np.random.random_sample((batch_size, 1)) * 0.2

        print('Training')

        for l in range(1, epochs + 1):
            for m in tqdm(range(1, self.datagenerator.__len__() + 1)):
                # Select images
                side_image, front_image = self.datagenerator.__getitem__(l - 1)

                frontalization_generated_image = self.frontalization_generator.predict(side_image)

                resolution_generated_image = self.resolution_generator.predict(frontalization_generated_image)

                self.resolution_discriminator.trainable = True

                # Train the discriminator (real classified as ones and generated as zeros)
                resolution_discriminator_fake_loss = self.resolution_discriminator.train_on_batch(resolution_generated_image, fake)
                resolution_discriminator_real_loss = self.resolution_discriminator.train_on_batch(front_image, real)
                resolution_discriminator_loss = 0.5 * np.add(resolution_discriminator_fake_loss, resolution_discriminator_real_loss)
                
                self.resolution_discriminator.trainable = False

                # Train the generator (wants discriminator to mistake images as real)
                resolution_generator_loss = self.resolution_gan.train_on_batch(frontalization_generated_image, [front_image, real])

                # Plot the progress
                print ('\nTraining epoch : %d \nTraining batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f ' 
                        % (l, m, resolution_discriminator_loss[1] * 100, resolution_discriminator_loss[0], resolution_generator_loss[2]))

                record = (l, m, resolution_discriminator_loss[1] * 100, resolution_discriminator_loss[0], resolution_generator_loss[2])
                self.history.append(record)

            self.datagenerator.on_epoch_end()

            # Save generated images and .h5
            if l % save_interval == 0:
                self.save_image(front_image = front_image, side_image = side_image, epoch_number = l, batch_number = m, save_path = self.save_path)

                # Check folder presence
                if not os.path.isdir(self.save_path + 'H5/'):
                    os.makedirs(self.save_path + 'H5/')

                self.resolution_generator.save(self.save_path + 'H5/' + 'resolution_epoch_%d.h5' % l)
                self.resolution_generator.save_weights(self.save_path + 'H5/' + 'resolution_weights_epoch_%d.h5' % l)

        self.history = np.array(self.history)

        self.graph(history = self.history, save_path = self.save_path + 'History/')

    def save_image(self, front_image, side_image, epoch_number, batch_number, save_path):
        # Rescale images 0 - 1
        # generated_image = 0.5 * self.generator.predict(side_image) + 0.5
        frontalization_generated_image = self.frontalization_generator.predict(side_image)

        resolution_generated_image = 0.5 * self.resolution_generator.predict(frontalization_generated_image) + 0.5

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        # Show image (first column : original side image, second column : original front image, third column : generated image(front image))
        for m in range(batch_size):
            plt.figure(figsize = (8, 2))

            # Adjust the interval of the image
            plt.subplots_adjust(wspace = 0.6)

            for n in range(self.n_show_image):
                generated_image_plot = plt.subplot(1, 3, n + 1 + (2 * self.n_show_image))
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(resolution_generated_image[m,  :  ,  :  ,  : ])

                original_front_face_image_plot = plt.subplot(1, 3, n + 1 + self.n_show_image)
                original_front_face_image_plot.set_title('Origninal front image')
                plt.imshow(front_image[m])

                original_side_face_image_plot = plt.subplot(1, 3, n + 1)
                original_side_face_image_plot.set_title('Origninal side image')
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

            save_name = 'Train%d_Batch%d_%d.png' % (epoch_number, batch_number, self.number)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()

        self.number = 1

    def graph(self, history, save_path):
        plt.plot(self.history[:, 2])     
        plt.plot(self.history[:, 3])
        plt.plot(self.history[:, 4])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generative adversarial network')
        plt.legend(['Accuracy of discriminator', 'Loss of discriminator', 'Loss of generator'], loc = 'upper left')

        figure = plt.gcf()   
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
    gan = GAN()
    gan.train(epochs = epochs, batch_size = batch_size, save_interval = save_interval)