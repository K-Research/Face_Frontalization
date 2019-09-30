from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, LeakyReLU, Reshape, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

# Generator
class Generator(object):
    def __init__(self, width = 28, height = 28, channels = 1, latent_size = 100, model_type = 'simple'):
        # Place a code here that initializes the class variables.
        self.W = width
        self.H = height
        self.C = channels
        self.LATENT_SPACE_SIZE = latent_size
        self.latent_space = np.random.normal(0, 1, (self.LATENT_SPACE_SIZE, ))

        if model_type == 'simple':
            # Initialize the simple generator here.
            self.Generator = self.model()
            self.OPTIMIZER = Adam(lr = 0.0002, decay = 8e-9)
            self.Generator.compile(loss = 'binary_crossentropy', optimizer = self.OPTIMIZER)

        elif model_type == 'DCGAN':
            # Initialize the convolution generator here.
            self.Generator = self.dc_model()
            self.OPTIMIZER = Adam(lr = 1e-4, beta_1 = 0.2)
            self.Generator.compile(loss = 'binary_crossentropy', optimizer = self.OPTIMIZER, metrics = ['accuracy'])

            self.save_model()
            self.summary()

    def dc_model(self):
        # Place the generator configured by the convolution method.
        model = Sequential()
        model.add(Dense(256 * 8 * 8, activation = LeakyReLU(0.2), input_dim = self.LATENT_SPACE_SIZE))
        model.add(BatchNormalization())

        model.add(Reshape((8, 8, 256)))
        model.add(UpSampling2D())
        
        # 16 x 16
        model.add(Conv2D(128, kernel_size = (5, 5), padding = 'same', activation = LeakyReLU(0.2)))
        model.add(BatchNormalization())
        model.add(UpSampling2D())

        # 3 x 64 x 64
        model.add(Conv2D(self.C, kernel_size = (5, 5), padding = 'same', activation = 'tanh'))
        return model
    
    def model(self, block_starting_size = 128, num_blocks = 4):
        # Place the simple GAN.
        return model
'''
    def summary(self):
        # Place the code here responsible for the function of the summary helper.
    
    def save_model(self):
        # Place the code here responsible for the function of the model saving helper.
'''
# Discriminator
class Discriminator(object):
    def __init__(self, width = 28, heigt = 28, channels = 1, latent_size = 100, model_type = 'simple'):
        # Place a code here that initializes the class variables.
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width * height * channels
        self.SHAPE = (width, height, channels)

        if model_type == 'simple':
            # Initialize the simple generator here.
            self.Discriminator = self.model()
            self.OPTIMIZER = Adam(lr = 0.0002, decay = 8e-9)
            self.Discriminator.compile(loss = 'binary_crossentropy', optimizer = self.OPTIMIZER, metrics = ['accuracy'])

        elif model_type == 'DCGAN':
            # Initialize the convolution generator here.
            self.Discriminator = self.dc_model()
            self.OPTIMIZER = Adam(lr = 1e-4, beta_1 = 0.2)
            self.Discriminator.compile(loss = 'binary_crossentropy', optimizer = self.OPTIMIZER, metrics = ['accuracy'])

            self.save_model()
            self.summary()

    def dc_model(self):
        # Place the generator configured by the convolution method.
        model = Sequential()
        model.add(Conv2D(64, kernel_size = (5, 5), stride = (2, 2), input_shape = (self.W, self.H, self.C), padding = 'same', activation = LeakyReLU(alpha = 0.2)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(128, kernel_size = (5, 5), stride = (2, 2), padding = 'same', activation = LeakyReLU(alpha = 0.2)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))
        return model
    
    def model(self, block_starting_size = 128, num_blocks = 4):
        # Place the simple GAN.
        return model
'''
    def summary(self):
        # Place the code here responsible for the function of the summary helper.
    
    def save_model(self):
        # Place the code here responsible for the function of the model saving helper.
'''
class Trainer:
    def __init__(self, width = 28, heigt = 28, channels = 1, latent_size = 100, epochs = 50000, batch = 32, checkpoint = 50, model_type = 'DCGAn', data_path = ''):
        self.generator = Generator(height = self.H, width = self.W, channels = self.C, latent_size = self.LATENT_SPACE_SIZE, model_type = self.model_type)
        self.discriminator = Discriminator(height = self.H, width = self.W, channels = self.C, model_type = self.model_type)
        self.gan = GAN(generator = self.generator.Generator, discriminator = self.discriminator.Discriminator)
        #self.load_MNIST()
        self.load_npy(data_path)

    def load_npy(self, npy_path, amount_of_data = 0.25):
        self.X_train = np.load(npy_path)
        self.X_train = self.X_train[ : int(amount_of_data * float(len(self.X_train)))]
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_train = np.expand_dimas(self.X_train, axis = 3)
        return
    
    def train(self):
        for e in range(self.EPOCHS):
            b = 0

            X_train_temp = deepcopy(self.X_train)
            while len(X_train_temp) > self.EPOCHS:
                # Follow the batch.
                b = b + 1

            # Train the discriminator.
            # The training batch for this model consists of half the real data and half the noise.
            # Grab the real image for this training batch.
            if self.flipCoin():
                count_real_images = int(self.BATCH)
                starting_index = randint(0, (len(X_train_temp) - count_real_images))
                real_images_raw = X_train_temp[starting_index : (starting_index + count_real_images)]
                # self.plot_check_batch(b, real_images_raw) # Send a lot of files.

                # Delete used images until no images remain.
                X_train_temp = np.delete(X_train_temp, range(starting_index, starting_index + count_real_images), 0)
                x_batch = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
                y_batch = np.ones([count_real_images, 1])
            
            else:
                # Grab the generated image for this instruction batch.
                latent_space_samples = self.sample_latent_space(self.BATCH)
                x_batch = self.generator.Generator.predict(latent_space_samples)
                y_batch = np.zeros([self.BATCH, 1])

            # Now use this batch to train the discriminator.
            discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

            # In fact, flipping the label when training the generator improves convergence.
            if self.flip(chance = 0.9):
                y_generated_labels = np.ones([self.BATCH, 1])
            
            else:
                y_generated_labels = np.zeros([self.BATCH, 1])

            X_latant_space_samples = self.sample_latent_space(self.BATCH)
            generator_loss = self.gan.gan_model.train_on_batch(X_latant_space_samples, y_generated_labels)

            print('Batch : ' + str(int(b)) +', [Discriminator :: Loss : '+ str(discriminator_loss) +'], [Generator :: Loss : '+ str(generator_loss) +']')

            if b % self.CHECKPOINT ==0:
                label = str(e) + '_' + str(b)
                self.plot_checkpoint(label)

            print('Epoch : ' + str(int(e)) +', [Discriminator :: Loss : '+ str(discriminator_loss) +'], [Generator :: Loss : '+ str(generator_loss) +']')

            if e % self.CHECKPOINT == 0:  
                self.plot_checkpoint(e)

            return