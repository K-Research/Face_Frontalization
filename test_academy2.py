from __future__ import print_function, division

from datagenerator_read_dir_face import DataGenerator, DataGenerator_predict
from glob import glob
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, Reshape, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

time = 3

# Load data
X_train = glob('D:/Korean 224X224X3 filtering/X/*jpg')
Y_train = glob('D:/Korean 224X224X3 filtering/Y/*jpg')

X_test = glob('D:/Test image/*jpg')

epochs = 100
batch_size = 32
save_interval = 1

class Autoencoder():
    def __init__(self):
        # Load data
        self.datagenerator = DataGenerator(X_train, Y_train, batch_size = batch_size)
        self.datagenerator_predict = DataGenerator_predict(X_test, batch_size = batch_size)

        # Prameters
        self.height = 224
        self.width = 224
        self.channels = 3

        self.optimizer = Adam(lr = 0.001, beta_1 = 0.5, beta_2 = 0.99)

        self.vgg16 = self.build_vgg16()

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 1
        self.save_path = 'D:/Generated Image/Training' + str(time) + '/'

        # Build and compile the autoencoder
        self.autoencoder = self.build_autoencoder()
        self.autoencoder.compile(loss = 'mse', optimizer = self.optimizer)

        # Save .json
        generator_model_json = self.autoencoder.to_json()

        # Check folder presence
        if not os.path.isdir(self.save_path + 'Json/'):
            os.makedirs(self.save_path + 'Json/')

        with open(self.save_path + 'Json/generator_model.json', "w") as json_file : 
            json_file.write(generator_model_json)

    def build_vgg16(self):
        vgg16 = VGGFace(include_top = False, model = 'vgg16', weights = 'vggface', input_shape = (self.height, self.width, self.channels))
        # Make trainable as False

        vgg16.trainable = False

        for layer in vgg16.layers:
            layer.trainable = False

        # vgg16.summary()

        return vgg16

    def build_autoencoder(self):
        generator_input = self.vgg16.get_layer('pool5').output

        generator_layer = Conv2DTranspose(filters = 1024, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_input)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'valid')(generator_layer)
        generator_layer = BatchNormalization(momentum = 0.8)(generator_layer)
        generator_layer = LeakyReLU(alpha = 0.2)(generator_layer)
        generator_layer = Conv2DTranspose(filters = self.channels, kernel_size = (5, 5), strides = (1, 1), padding = 'valid')(generator_layer)

        generator_output = Activation('tanh')(generator_layer)

        generator = Model(inputs = self.vgg16.input, outputs = generator_output)

        # generator.summary()

        return generator

    def train(self, epochs, batch_size, save_interval):
        print('Training')

        for k in range(1, epochs + 1):
            for l in tqdm(range(1, self.datagenerator.__len__() + 1)):
                # Select images
                side_image, front_image = self.datagenerator.__getitem__(l - 1)
                
                # Train the autoencoder (real classified as ones and generated as zeros)
                autoencoer_loss = self.autoencoder.train_on_batch(side_image, front_image)
                
                # Plot the progress
                print ('\nTraining epoch : %d \nTraining batch : %d \nLoss of autoencoder : %f ' % (k, l, autoencoer_loss))

                record = (k, l, autoencoer_loss)

                self.history.append(record)

            # If at save interval -> save generated image samples
            if k % save_interval == 0:
                test_image = self.datagenerator_predict.__getitem__(l - 1)        
                self.save_test_image(side_image = test_image, epoch_number = k, batch_number = l, save_path = self.save_path)

            self.datagenerator.on_epoch_end()

            # Save .h5
            if k % save_interval == 0:
                # Check folder presence
                if not os.path.isdir(self.save_path + 'H5/'):
                    os.makedirs(self.save_path + 'H5/')

                self.autoencoder.save(self.save_path + 'H5/' + 'generator_epoch_%d.h5' % k)
                self.autoencoder.save_weights(self.save_path + 'H5/' + 'generator_weights_epoch_%d.h5' % k)

        self.history = np.array(self.history)

        self.graph(history = self.history, save_path = self.save_path + 'History/')

    def save_train_image(self, front_image, side_image, epoch_number, batch_number, save_path):
        # Rescale images 0 - 1
        generated_image = 0.5 * self.autoencoder.predict(side_image) + 0.5

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

    def save_test_image(self, side_image, epoch_number, batch_number, save_path):
        # Rescale images 0 - 1
        generated_image = 0.5 * self.autoencoder.predict(side_image) + 0.5

        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        # Show image (first column : original side image, second column : generated image(front image))
        for m in range(batch_size):
            plt.figure(figsize = (8, 2))

            # Adjust the interval of the image
            plt.subplots_adjust(wspace = 0.6)

            for n in range(self.n_show_image):
                generated_image_plot = plt.subplot(1, 3, n + 1 + self.n_show_image)
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(generated_image[m,  :  ,  :  ,  : ])

                original_side_face_image_plot = plt.subplot(1, 3, n + 1)
                original_side_face_image_plot.set_title('Origninal side image')
                plt.imshow(side_image[m])

                # Don't show axis of x and y
                generated_image_plot.axis('off')
                original_side_face_image_plot.axis('off')

                self.number += 1

                # plt.show()

            save_path = save_path

            # Check folder presence
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_name = 'Test%d_Batch%d_%d.png' % (epoch_number, batch_number, self.number)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()

        self.number = 1

    def graph(self, history, save_path):
        plt.plot(self.history[:, 2])     
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
    autoencoder = Autoencoder()
    autoencoder.train(epochs = epochs, batch_size = batch_size, save_interval = save_interval)