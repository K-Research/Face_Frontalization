from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from datagenerator_read_dir_face import DataGenerator
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

height = 224
width = 224
channels = 3
z_dimension = 512
batch_size = 32
epochs = 1000
line = 101
n_show_image = 1
nb_class = 2
hidden_dim = 512
vgg = VGGFace(include_top=False, model="vgg16", input_shape=(224, 224, 3), weights= 'vggface')
vgg.trainable = False
vgg.summary()
optimizerD = Adam(lr = 0.00002, beta_1 = 0.5, beta_2 = 0.999)
optimizerC = Adam(lr = 0.002, beta_1 = 0.5, beta_2 = 0.999)
number = 0


X_train = glob('D:/Korean 224X224X3 filtering/X/*jpg')
Y_train = glob('D:/Korean 224X224X3 filtering/Y/*jpg')

# X = np.load("./swh/npy/X.npy")
# Y = np.load("./swh/npy/Y.npy")

# X = X / 127.5 - 1
# Y = Y / 127.5 - 1

class model_1():

    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.z_dimension = z_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.line = line
        self.n_show_image = n_show_image
        self.vgg = vgg
        self.optimizerD = optimizerD
        self.optimizerC = optimizerC
        self.DG = DataGenerator(X_train, Y_train, batch_size = batch_size)
        self.number = number
        self.history = []

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizerD, metrics = ['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        z = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(z)

        valid = self.discriminator(image)

        self.combined = Model(z, valid)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = self.optimizerC, metrics = ['accuracy'])

    def conv2d_block(self, layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
        input = layers

        layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input)
        layer = BatchNormalization(momentum = momentum)(layer)
        output = LeakyReLU(alpha = alpha)(layer)
        # output = ReLU()(layer)

        # model = Model(model, output)

        return output

    def deconv2d_block(self, layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
        input = layers

        layer = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(input)
        layer = BatchNormalization(momentum = momentum)(layer)
        output = LeakyReLU(alpha = alpha)(layer)
        # output = ReLU()(layer)

        # model = Model(model, output)

        return output

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size = (4, 4), strides = (2, 2), input_shape = (self.height, self.width, self.channels), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(32, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(64, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(128, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(256, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(512, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(1024, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(1, kernel_size = (4, 4), strides = (2, 2), padding = 'same'))
        model.add(BatchNormalization(momentum = 0.8))
        model.add(LeakyReLU(alpha = 0.2))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units = 1, activation = 'sigmoid'))

    def build_generator(self):
        # self.vgg.summary()
        input = self.vgg.get_layer("pool5").output
        layers = self.deconv2d_block(input, 512)
        layers = self.deconv2d_block(layers, 512)
        layers = self.deconv2d_block(layers, 256)
        layers = self.deconv2d_block(layers, 128)
        output = Conv2DTranspose(filters = 3, kernel_size = (4, 4), strides = 2, activation = 'tanh', padding = 'same')(layers)
        model = Model(self.vgg.input, output)
        for layer in model.layers:
            layer.trainable = False
            # print(layer.get_weights())
            if layer.name == "pool5":
                break
        # model.summary()
        return model

    
    def train(self, epochs, batch_size, save_interval):
        fake = np.zeros((batch_size))
        real = np.ones((batch_size))

        for i in range(epochs):
            for j in range(self.DG.__len__()):
            # for j in range(batch_size):
                # index = np.random.randint(0, X.shape[0], batch_size)
                # front_images = Y[index]
                # side_images = X[index]
                side_images, front_images = self.DG.__getitem__(j)

                generated_images = self.generator.predict(side_images)

                discriminator_fake_loss = self.discriminator.train_on_batch(generated_images, fake)
                discriminator_real_loss = self.discriminator.train_on_batch(front_images, real)
                discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss) * 0.5
                
                generator_loss = self.combined.train_on_batch(side_images, real)

                print ('\nTraining epoch : %d \nTraining batch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nAccuracy of generator : %.2f%%  \nLoss of generator : %f'
                        % (i + 1, j + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[1] * 100, generator_loss[0]))
                
                record = (i + 1, j + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss[1] * 100, generator_loss[0])
                self.history.append(record)

                # if j % save_interval == 0:
                #     save_path = 'D:/Generated Image/Training' + str(line) + '/'
                #     self.save_image(epoch = i, batch = j, front_image = front_images, side_image = side_images, save_path = save_path)

            self.DG.on_epoch_end()
            if i % 1 == 0:
                self.generator.save("D:/Generated Image/Training/{1}_{0}.h5".format(str(i), str(line)))
                save_path = 'D:/Generated Image/Training' + str(line) + '/'
                self.save_image(epoch = i, batch = j, front_image = front_images, side_image = side_images, save_path = save_path)

        self.history = np.array(self.history)

        self.graph(history = history, save_path = save_path)

    def save_image(self, epoch, batch, front_image, side_image, save_path):
        # Rescale images 0 - 1
        # generated_image = (255 * ((self.generator.predict(side_image) + 1)/2)).astype(np.uint8)
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5

        # print("generated_image.shape :", generated_image.shape)
        front_image = (255 * ((front_image) + 1)/2).astype(np.uint8)
        # print("front_image.shape :", front_image.shape)        
        side_image = (255 * ((side_image)+1)/2).astype(np.uint8)
        # print("side_image.shape :", side_image.shape)

        
        for i in range(self.batch_size):
            plt.figure(figsize = (8, 2))

            # Adjust the interval of the image
            plt.subplots_adjust(wspace = 0.6)

            # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
            for m in range(n_show_image):
                generated_image_plot = plt.subplot(1, 3, m + 1 + (2 * n_show_image))
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(generated_image[i])

                original_front_face_image_plot = plt.subplot(1, 3, m + 1 + n_show_image)
                original_front_face_image_plot.set_title('Origninal front image')
                plt.imshow(front_image[i])

                original_side_face_image_plot = plt.subplot(1, 3, m + 1)
                original_side_face_image_plot.set_title('Origninal side image')
                plt.imshow(side_image[i])

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

            save_name = '%d-%d-%d.png' % (epoch, batch, i)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()

    def graph(self, history, save_path):
        plt.plot(history[:, 2])     
        plt.plot(history[:, 3])
        plt.plot(history[:, 4])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generative adversarial network')
        plt.legend(['Accuracy of discriminator', 'Loss of discriminator', 'Loss of generator'], loc = 'upper left')

        figure = plt.gcf()

        plt.show()

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
    dcgan = model_1()
    dcgan.train(epochs = epochs, batch_size = batch_size, save_interval = n_show_image)