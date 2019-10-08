from __future__ import print_function, division

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.layers import Activation, add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, Nadam
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
from tqdm import tqdm

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

train_epochs = 100000
test_epochs = 1
train_batch_size = 4
test_batch_size = 1
train_save_interval = 1
test_save_interval = 1

class DCGAN():
    def __init__(self):
        # Rescale -1 to 1
        self.X_train = X_train / 127.5 - 1.
        self.Y_train = Y_train / 127.5 - 1.
        # self.X_test = X_test / 127.5 - 1.
        # self.Y_test = Y_test / 127.5 - 1.

        # Prameters
        self.height = self.X_train.shape[1]
        self.width = self.X_train.shape[2]
        self.channels = self.X_train.shape[3]
        self.latent_dimension = self.width

        self.optimizer = Adam(lr = 0.0002, beta_1 = 0.5)

        self.n_show_image = 1 # Number of images to show
        self.history = []
        self.number = 0

        # Build and compile the discriminator
        # self.discriminator = self.build_discriminator()
        self.discriminator_A = self.build_discriminator() # Modify
        self.discriminator_B = self.build_discriminator() # Modify
        # self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])
        self.discriminator_A.compile(loss = 'mse', optimizer = self.optimizer, metrics = ['accuracy']) # Modify
        self.discriminator_B.compile(loss = 'mse', optimizer = self.optimizer, metrics = ['accuracy']) # Modify

        # Build and compile the generator
        # self.generator = self.build_generator()
        # self.generator.compile(loss = self.vgg19_loss, optimizer = self.optimizer)

        self.generator_AB = self.build_generator() # Modify
        self.generator_BA = self.build_generator() # Modify

        # # The generator takes noise as input and generates imgs
        # z = Input(shape = (self.height, self.width, self.channels))
        # image = self.generator(z)

        # Input images from both domains
        image_A = Input(shape = (self.height, self.width, self.channels)) # Modify
        image_B = Input(shape = (self.height, self.width, self.channels)) # Modify

        # Translate images to the other domain
        fake_A = self.generator_AB(image_B) # Modify
        fake_B = self.generator_BA(image_A) # Modify

        # Translate images back to original domain
        reconstructure_A = self.generator_BA(fake_B)
        reconstructure_B = self.generator_BA(fake_A)

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False
        self.discriminator_A.trainable = False # Modify
        self.discriminator_B.trainable = False # Modify


        # The discriminator takes generated images as input and determines validity
        # valid = self.discriminator(image)
        valid_A = self.discriminator_A(fake_A) # Modify
        valid_B = self.discriminator_A(fake_B) # Modify

        # # The combined model  (stacked generator and discriminator)
        # # Trains the generator to fool the discriminator
        # self.combined = Model(z, [image, valid])
        self.combined = Model(inputs = [image_A, image_B], outputs = [valid_A, valid_B, fake_B, fake_A, reconstructure_A, reconstructure_B]) # Modify
        # self.combined.compile(loss = [self.vgg19_loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer = self.optimizer)
        self.combined.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], optimizer = self.optimizer) # Modify


        # self.combined.summary()

    def discriminator_block(self, model, filters, kernel_size, strides):
        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(model)
        layer = BatchNormalization(momentum = 0.5)(layer)
        layer = LeakyReLU(alpha = 0.2)(layer)

        return layer

    def residual_block(self, model, filters, kernel_size, strides):
        generator = model

        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(generator)
        layer = BatchNormalization(momentum = 0.5)(layer)

        # Using Parametric ReLU
        layer = PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = [1, 2])(layer)
        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = 'same')(layer)
        output = BatchNormalization(momentum = 0.5)(layer)

        model = add([generator, output])

        return model

    def up_sampling_block(self, model, filters, kernel_size, strides):
        # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
        # Even we can have our own function for deconvolution (i.e one made in Utils.py)
        # layer = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = 'same)(layer)
        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(model)
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
        input = Input(shape = (self.height, self.width, self.channels)) # Modify

        downsampling_layer1 = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(input) # Modify
        downsampling_layer1 = LeakyReLU(alpha = 0.2)(downsampling_layer1) # Modify
        downsampling_layer2 = Conv2D(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer1) # Modify
        downsampling_layer2 = LeakyReLU(alpha = 0.2)(downsampling_layer2) # Modify
        # downsampling_layer2 = InstanceNormalization()(downsampling_layer2) # Modify
        downsampling_layer3 = Conv2D(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer2) # Modify
        downsampling_layer3 = LeakyReLU(alpha = 0.2)(downsampling_layer3) # Modify
        # downsampling_layer3 = InstanceNormalization()(downsampling_layer3) # Modify
        downsampling_layer4 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer3) # Modify
        downsampling_layer4 = LeakyReLU(alpha = 0.2)(downsampling_layer4) # Modify
        # downsampling_layer4 = InstanceNormalization()(downsampling_layer4) # Modify
        downsampling_layer5 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer4) # Modify
        downsampling_layer5 = LeakyReLU(alpha = 0.2)(downsampling_layer5) # Modify
        # downsampling_layer5 = InstanceNormalization()(downsampling_layer5) # Modify
        downsampling_layer6 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer5) # Modify
        downsampling_layer6 = LeakyReLU(alpha = 0.2)(downsampling_layer6) # Modify
        # downsampling_layer6 = InstanceNormalization()(downsampling_layer6) # Modify
        downsampling_layer7 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(downsampling_layer6) # Modify
        downsampling_layer7 = LeakyReLU(alpha = 0.2)(downsampling_layer7) # Modify
        # downsampling_layer7 = InstanceNormalization()(downsampling_layer7) # Modify

        upsampling_layer1 = UpSampling2D(size = (2, 2))(downsampling_layer7) # Modify
        upsampling_layer1 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer1) # Modify
        # upsampling_layer1 = InstanceNormalization()(upsampling_layer1) # Modify
        upsampling_layer1 = Concatenate()([upsampling_layer1, downsampling_layer6]) # Modify
        upsampling_layer2 = UpSampling2D(size = (2, 2))(upsampling_layer1) # Modify
        upsampling_layer2 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer2) # Modify
        # upsampling_layer2 = InstanceNormalization()(upsampling_layer2) # Modify
        upsampling_layer2 = Concatenate()([upsampling_layer2, downsampling_layer5]) # Modify
        upsampling_layer3 = UpSampling2D(size = (2, 2))(upsampling_layer2) # Modify
        upsampling_layer3 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer3) # Modify
        # upsampling_layer3 = InstanceNormalization()(upsampling_layer3) # Modify
        upsampling_layer3 = Concatenate()([upsampling_layer3, downsampling_layer4]) # Modify
        upsampling_layer4 = UpSampling2D(size = (2, 2))(upsampling_layer3) # Modify
        upsampling_layer4 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer4) # Modify
        # upsampling_layer4 = InstanceNormalization()(upsampling_layer4) # Modify
        upsampling_layer4 = Concatenate()([upsampling_layer4, downsampling_layer3]) # Modify
        upsampling_layer5 = UpSampling2D(size = (2, 2))(upsampling_layer4) # Modify
        upsampling_layer5 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer5) # Modify
        # upsampling_layer5 = InstanceNormalization()(upsampling_layer5) # Modify
        upsampling_layer5 = Concatenate()([upsampling_layer5, downsampling_layer2]) # Modify
        upsampling_layer6 = UpSampling2D(size = (2, 2))(upsampling_layer5) # Modify
        upsampling_layer6 = Conv2D(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same')(upsampling_layer6) # Modify
        # upsampling_layer6 = InstanceNormalization()(upsampling_layer6) # Modify
        upsampling_layer6 = Concatenate()([upsampling_layer6, downsampling_layer1]) # Modify
        upsampling_layer7 = UpSampling2D(size = (2, 2))(upsampling_layer6) # Modify

        output = Conv2D(filters = self.channels, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'tanh')(upsampling_layer7) # Modify

        generator_model = Model(input, output) # Modify

        # generator_model.summary() # Modify

        return generator_model # Modify

    def build_discriminator(self):
        input = Input(shape = (self.height, self.width, self.channels)) # Modify

        layer = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(input) # Modify
        layer = LeakyReLU(alpha = 0.2)(layer) # Modify
        layer = Conv2D(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer) # Modify
        layer = LeakyReLU(alpha = 0.2)(layer) # Modify
        # layer = InstanceNormalization()(layer) # Modify
        layer = Conv2D(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer) # Modify
        layer = LeakyReLU(alpha = 0.2)(layer) # Modify
        # layer = InstanceNormalization()(layer) # Modify
        layer = Conv2D(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(layer) # Modify
        layer = LeakyReLU(alpha = 0.2)(layer) # Modify
        # layer = InstanceNormalization()(layer) # Modify
        layer = Flatten()(layer)

        output = Dense(units = 1, activation = 'sigmoid')(layer)
        
        discriminator_model = Model(input, output) # Modify

        # discriminator_model.summary() # Modify

        return discriminator_model # Modify

    def train(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        fake = np.zeros((batch_size, 1)) # Modify
        real = np.ones((batch_size, 1)) # Modify

        print('Training')

        for k in range(1, epochs + 1):
            for l in tqdm(range(batch_size)):
            # for l, (image_a, image_b) in tqdm(range(batch_size)): # Modify
                # Select a random half of images
                index = np.random.randint(0, self.X_train.shape[0], batch_size)
                front_image = self.Y_train[index]

                # Generate a batch of new images
                side_image = self.X_train[index]

                # optimizer.zero_grad()
                
                generated_image_A = self.generator_BA.predict(front_image) # Modify
                generated_image_B = self.generator_AB.predict(side_image) # Modify


                # Train the discriminator (real classified as ones and generated as zeros)
                discriminator_A_fake_loss = self.discriminator_A.train_on_batch(generated_image_A, fake) # Modify
                discriminator_A_real_loss = self.discriminator_A.train_on_batch(side_image, real) # Modify
                discriminator_A_loss = 0.5 * np.add(discriminator_A_fake_loss, discriminator_A_real_loss) # Modify

                discriminator_B_fake_loss = self.discriminator_B.train_on_batch(generated_image_B, fake) # Modify
                discriminator_B_real_loss = self.discriminator_B.train_on_batch(front_image, real) # Modify
                discriminator_B_loss = 0.5 * np.add(discriminator_B_fake_loss, discriminator_B_real_loss) # Modify

                discriminator_loss = 0.5 * np.add(discriminator_A_loss, discriminator_B_loss)
                
                # Train the generator (wants discriminator to mistake images as real)
                generator_loss = self.combined.train_on_batch([side_image, front_image], [real, real, front_image, side_image, side_image, front_image]) # Modify

                # Plot the progress
                print ('\nTraining epoch : %d \nTraining batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f ' 
                        % (k, l, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2]))

                record = (k, l, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2])
                self.history.append(record)

                # If at save interval -> save generated image samples
                if l % save_interval == 0:
                    save_path = 'D:/Generated Image/Training' + str(time) + '/'
                    self.save_image(image_index = l, front_image = front_image, side_image = side_image, save_path = save_path)

            if k % 100 == 0:
                self.generator_AB.save(save_path + 'generator_AB_epoch_%d.h5' % k)
                self.generator_BA.save(save_path + 'generator_BA_epoch_%d.h5' % k)

        self.history = np.array(self.history)

        self.graph(history = self.history, save_path = save_path)

    def test(self, epochs, batch_size, save_interval):
        # Adversarial ground truths
        fake = np.zeros((batch_size, 1))
        real = np.ones((batch_size, 1))

        print('Testing')

        for m in range(1, epochs + 1):
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
                        % (m, n, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2]))

                record = (m, n, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[2])
                history.append(record)

                # If at save interval -> save generated image samples
                if n % save_interval == 0:
                    save_path = 'D:/Generated Image/Testing' + str(time) + '/'
                    self.save_image(image_index = n, front_image = front_image, side_image = side_image, save_path = save_path)

        history = np.array(history)

        self.history(history = history, save_path = save_path)

    def save_image(self, image_index, front_image, side_image, save_path):
        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator_AB.predict(side_image) + 0.5 # Modify

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        plt.figure(figsize = (8, 2))

        # Adjust the interval of the image
        plt.subplots_adjust(wspace = 0.6)

        # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
        for m in range(self.n_show_image):
            generated_image_plot = plt.subplot(1, 3, m + 1 + (2 * self.n_show_image))
            generated_image_plot.set_title('Generated image (front image)')

            if self.channels == 1:
                plt.imshow(generated_image[image_index,  :  ,  :  , 0], cmap = 'gray')
            
            else:
                plt.imshow(generated_image[image_index,  :  ,  :  ,  : ])

            original_front_face_image_plot = plt.subplot(1, 3, m + 1 + self.n_show_image)
            original_front_face_image_plot.set_title('Origninal front image')

            if self.channels == 1:
                plt.imshow(front_image[image_index].reshape(self.height, self.width), cmap = 'gray')
                
            else:
                plt.imshow(front_image[image_index])

            original_side_face_image_plot = plt.subplot(1, 3, m + 1)
            original_side_face_image_plot.set_title('Origninal side image')

            if self.channels == 1:
                plt.imshow(side_image[image_index].reshape(self.height, self.width), cmap = 'gray')
                
            else:
                plt.imshow(side_image[image_index])

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

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs = train_epochs, batch_size = train_batch_size, save_interval = train_save_interval)
    # dcgan.test(epochs = test_epochs, batch_size = test_batch_size, save_interval = test_save_interval)