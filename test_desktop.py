from __future__ import print_function, division

from datagenerator_read_dir_face import DataGenerator
from glob import glob
from keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, Reshape, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

time = 11

# Load data
X_train = glob('D:/Bitcamp/Project/Frontalization/Imagenius/Data/Korean 224X224X3 filtering/X/*jpg')
Y_train = glob('D:/Bitcamp/Project/Frontalization/Imagenius/Data/Korean 224X224X3 filtering/Y/*jpg')

epochs = 100
batch_size = 64
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

        self.vgg16 = self.build_vgg16()
        self.resnet50 = self.build_resnet50()

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 1
        self.save_path = 'D:/Generated Image/Training' + str(time) + '/'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()

        # Save .json
        generator_model_json = self.generator.to_json()

        # Check folder presence
        if not os.path.isdir(self.save_path + 'Json/'):
            os.makedirs(self.save_path + 'Json/')

        with open(self.save_path + 'Json/generator_model.json', "w") as json_file : 
            json_file.write(generator_model_json)

        # The generator takes noise as input and generates imgs
        z = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validiy
        valid = self.discriminator(image)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, [image, valid])
        self.combined.compile(loss = 'binary_crossentropy', optimizer = self.optimizer)

        # self.combined.summary()

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
        
    def build_vgg16(self):
        vgg16 = VGGFace(include_top = False, model = 'vgg16', weights = 'vggface', input_shape = (self.height, self.width, self.channels))

        # Make trainable as False

        vgg16.trainable = False

        for i in vgg16.layers:
            i.trainable = False

        # vgg16.summary()

        return vgg16

    def build_resnet50(self):
        resnet50 = VGGFace(include_top = False, model = 'resnet50', weights = 'vggface', input_shape = (self.height, self.width, self.channels))

        # Make trainable as False

        resnet50.trainable = False

        for j in resnet50.layers:
            j.trainable = False

        # resnet50.summary()

        return resnet50

    def build_generator(self):
        generator_input = self.vgg16.get_layer('pool5').output

        # residual_input = self.vgg16.get_layer('conv3_3').output

        generator_layer = Conv2DTranspose(filters = 1024, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(generator_input)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)

        # generator_layer = add([residual_input, generator_layer])

        generator_layer = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = self.channels, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(generator_layer)

        generator_output = Activation('tanh')(generator_layer)

        generator = Model(inputs = self.vgg16.input, outputs = generator_output)

        # generator.summary()

        return generator

    def build_discriminator(self):
        # discriminator_input = Input(shape = (self.height, self.width, self.channels))
        
        # discriminator_layer = Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')(discriminator_input)
        # discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        # discriminator_layer = Dropout(rate = 0.25)(discriminator_layer)
        # discriminator_layer = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')(discriminator_layer)
        # discriminator_layer = BatchNormalization(momentum = 0.8)(discriminator_layer)
        # discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        # discriminator_layer = Dropout(rate = 0.25)(discriminator_layer)
        # discriminator_layer = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')(discriminator_layer)
        # discriminator_layer = BatchNormalization(momentum = 0.8)(discriminator_layer)
        # discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        # discriminator_layer = Dropout(rate = 0.25)(discriminator_layer)
        # discriminator_layer = Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')(discriminator_layer)
        # discriminator_layer = BatchNormalization(momentum = 0.8)(discriminator_layer)
        # discriminator_layer = LeakyReLU(alpha = 0.2)(discriminator_layer)
        # discriminator_layer = Dropout(rate = 0.25)(discriminator_layer)
        # discriminator_layer = Flatten()(discriminator_layer)

        discriminator_input = self.resnet50.get_layer('avg_pool').output

        discriminator_layer = Flatten()(discriminator_input)
        
        discriminator_output = Dense(units = 1, activation = 'sigmoid')(discriminator_layer)

        discriminator = Model(inputs = self.resnet50.output, outputs = discriminator_output)

        # discriminator.summary()

        return discriminator

    def train(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        # fake = np.zeros((batch_size, 1))
        # real = np.ones((batch_size, 1))
        fake = np.random.random_sample((batch_size, 1)) * 0.2
        real = np.ones((batch_size, 1)) - np.random.random_sample((batch_size, 1)) * 0.2

        print('Training')

        for k in range(1, epochs + 1):
            for l in tqdm(range(1, self.datagenerator.__len__() + 1)):
                # Select images
                side_image, front_image = self.datagenerator.__getitem__(k - 1)

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

                # # If at save interval -> save generated image samples
                # if l % save_interval == 0:
                #     self.save_image(front_image = front_image, side_image = side_image, epoch_number = k, batch_number = l, save_path = self.save_path)

            self.datagenerator.on_epoch_end()

            # Save generated images and .h5
            if k % save_interval == 0:
                self.save_image(front_image = front_image, side_image = side_image, epoch_number = k, batch_number = l, save_path = self.save_path)

                # Check folder presence
                if not os.path.isdir(self.save_path + 'H5/'):
                    os.makedirs(self.save_path + 'H5/')

                self.generator.save(self.save_path + 'H5/' + 'generator_epoch_%d.h5' % k)
                self.generator.save_weights(self.save_path + 'H5/' + 'generator_weights_epoch_%d.h5' % k)

        self.history = np.array(self.history)

        self.graph(history = self.history, save_path = self.save_path + 'History/')

    def save_image(self, front_image, side_image, epoch_number, batch_number, save_path):
        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5

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
                plt.imshow(generated_image[m,  :  ,  :  ,  : ])

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