import numpy as np
from keras.layers import Conv2D, Deconv2D, LeakyReLU, BatchNormalization, MaxPool2D
from keras.models import Model, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from datagenerator_read_dir import DataGenerator
from glob import glob


X_train = glob("D:/X_train/*jpg")
Y_train = glob("D:/Y_train/*jpg")

# X = X / 127.5 - 1
# Y = Y / 127.5 - 1

height = 128
width = 128
channels = 3
z_dimension = 512
batch_size = 32
epochs = 10000
line = 3
n_show_image = 1
DG = DataGenerator(X_train, Y_train, batch_size = batch_size)

optimizerD = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
optimizerG = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999)

history = []

def conv2d_block(layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
    input = layers

    layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input)
    layer = BatchNormalization(momentum = momentum)(layer)
    output = LeakyReLU(alpha = alpha)(layer)

    # model = Model(model, output)

    return output

def deconv2d_block(layers, filters, kernel_size = (4, 4), strides = 2, momentum = 0.8, alpha = 0.2):
    input = layers

    layer = Deconv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(input)
    layer = BatchNormalization(momentum = momentum)(layer)
    output = LeakyReLU(alpha = alpha)(layer)

    # model = Model(model, output)

    return output
    
class G():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.z_dimension = z_dimension
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.number = 0
        self.history = []

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizerD, metrics = ['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        z = Input(shape = (self.height, self.width, self.channels))
        image = self.generator(z)

        valid = self.discriminator(image)

        self.combined = Model(z, valid)
        self.combined.compile(loss = 'binary_crossentropy', optimizer = optimizerG)

    def build_discriminator(self):
        input = Input(shape = (self.height, self.width, self.channels))

        layers = conv2d_block(input, 16)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 32)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 64)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 128)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 256)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 512)
        # layers = MaxPool2D(2)(layers)       
        output = Conv2D(1, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'sigmoid')(layers)
        
        model = Model(input, output)
        model.summary()
        return model

    def build_generator(self):
        input = Input(shape = (self.height, self.width, self.channels))

        layers = conv2d_block(input, 16)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 32)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 64)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 128)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 256)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 512)
        # layers = MaxPool2D(2)(layers)
        layers = conv2d_block(layers, 512)        
        # layers = MaxPool2D(2)(layers)

        layers = deconv2d_block(layers, 512)
        layers = deconv2d_block(layers, 256)
        layers = deconv2d_block(layers, 128)
        layers = deconv2d_block(layers, 64)
        layers = deconv2d_block(layers, 32)
        layers = deconv2d_block(layers, 16)
        output = Deconv2D(filters = 3, kernel_size = (4, 4), strides = 2, activation = 'tanh', padding = 'same')(layers)
  
        model = Model(input, output)
        model.summary()
        return model

    def train(self, epochs, batch_size, save_interval):
        fake = np.zeros((batch_size, 1, 1, 1))
        real = np.ones((batch_size, 1, 1, 1))

        for i in range(epochs):
            # for j in range(batch_size):
                # index = np.random.randint(0, X.shape[0], batch_size)
                # front_images = Y[index]
                # side_images = X[index]
            side_images, front_images = DG.__getitem__(i)

            generated_images = self.generator.predict(side_images)

            discriminator_fake_loss = self.discriminator.train_on_batch(generated_images, fake)
            discriminator_real_loss = self.discriminator.train_on_batch(front_images, real)
            discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss) * 0.5
            
            discriminator_fake_loss += self.discriminator.train_on_batch(generated_images, fake)
            discriminator_real_loss += self.discriminator.train_on_batch(front_images, real)
            discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss) * 0.5

            generator_loss = self.combined.train_on_batch(side_images, real)

            print ('\nTraining epoch : %d \nAccuracy of discriminator : %.2f%% \nLoss of discriminator : %f \nLoss of generator : %f'
                    % (i + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss))
            
            # record = (m + 1, n + 1, discriminator_loss[1] * 100, discriminator_loss[0], generator_loss)
            # self.history.append(record)

            if i % save_interval == 0:
                save_path = 'D:/Generated Image/Training' + str(line) + '/'
                self.save_image(image_index = i, front_image = front_images, side_image = side_images, save_path = save_path)

            if i % 100 == 0:
                self.generator.save("./swh/model/ae+gan/{1}_{0}.h5".format(str(i), str(line)))

        # self.history = np.array(self.history)

        self.graph(history = history, save_path = save_path)

    def save_image(self, image_index, front_image, side_image, save_path):
        number = self.number

        # Rescale images 0 - 1
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5

        front_image = (127.5 * (front_image + 1)).astype(np.uint8)
        side_image = (127.5 * (side_image + 1)).astype(np.uint8)

        plt.figure(figsize = (8, 2))

        # Adjust the interval of the image
        plt.subplots_adjust(wspace = 0.6)

        # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
        for m in range(n_show_image):
            generated_image_plot = plt.subplot(1, 3, m + 1 + (2 * n_show_image))
            generated_image_plot.set_title('Generated image (front image)')

            if channels == 1:
                plt.imshow(generated_image[image_index,  :  ,  :  , 0], cmap = 'gray')
            
            else:
                plt.imshow(generated_image[image_index,  :  ,  :  ,  : ])

            original_front_face_image_plot = plt.subplot(1, 3, m + 1 + n_show_image)
            original_front_face_image_plot.set_title('Origninal front image')

            if channels == 1:
                plt.imshow(front_image[image_index].reshape(height, width), cmap = 'gray')
                
            else:
                plt.imshow(front_image[image_index])

            original_side_face_image_plot = plt.subplot(1, 3, m + 1)
            original_side_face_image_plot.set_title('Origninal side image')

            if channels == 1:
                plt.imshow(side_image[image_index].reshape(height, width), cmap = 'gray')
                
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

        save_name = '%d.png' % number
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
    dcgan = G()
    dcgan.train(epochs = epochs, batch_size = batch_size, save_interval = n_show_image)