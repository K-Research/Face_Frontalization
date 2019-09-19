from keras import backend as K
from keras import objectives
import keras.losses
from keras.layers import concatenate, Conv2D, Dense, Flatten, Input, Lambda, MaxPooling2D, Reshape, UpSampling2D
from keras.losses import mse
from keras.models import Model, Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data
X = np.load('./npy/cvaex60.npy') # Angular image with front
Y = np.load('./npy/cvaey60.npy') # Label of each angle

X_test = np.load('./npy/test.npy') # Two persons

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# print(X.shape) # (300, 60, 60)
# print(Y)
# print(Y.shape) # (300, )

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.1, random_state = 66)

# print(X_train.shape) # (270, 60, 60)
# print(Y_train.shape) # (30, 60, 60)
# print(X_validation.shape) # (270, )
# print(Y_validation.shape) # (30, )

# One-hot encoding
y_train = to_categorical(Y_train)
y_validation = to_categorical(Y_validation)

# print(y_train.shape) # (270, 10)
# print(y_validation.shape) # (30, 10)

# Parameters
batch_size = 64
epochs = 100
latent_dimension = 2
n_pixels = X_train.shape[1]
channel = X_train.shape[3]

leaky_relu = tf.nn.leaky_relu
parametric_relu = keras.layers.PReLU(alpha_initializer = 'zeros', alpha_regularizer = None, alpha_constraint = None, shared_axes = None)

# Encoder
encoder_input = Input(shape = (n_pixels, n_pixels, channel))
condition_input = Input(shape = (10, ), name = 'Label')

condition_layer = Dense(n_pixels * n_pixels)(condition_input)
condition_layer = Reshape((n_pixels, n_pixels, 1))(condition_layer)

concatenate_layer = concatenate([encoder_input, condition_layer])
encoder_layer = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(concatenate_layer)
encoder_layer = MaxPooling2D(pool_size = (2, 2))(encoder_layer)
encoder_layer = Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(encoder_layer)
encoder_layer = MaxPooling2D(pool_size = (2, 2))(encoder_layer)
encoder_layer = Conv2D(16, kernel_size = (3, 3), padding = 'same', activation = 'relu')(encoder_layer)

latent_shape = K.int_shape(encoder_layer) # The shape of tensor or variable as a tuple of int

encoder_layer = Flatten()(encoder_layer)
encoder_layer = Dense(n_pixels // 2, activation = 'relu')(encoder_layer)

latent_mean = Dense(latent_dimension)(encoder_layer)
latent_log_value = Dense(latent_dimension)(encoder_layer)

def sampling(args):
    latent_mean, laten_log_value = args
    batch = K.shape(latent_mean)[0]
    dimension = K.int_shape(latent_mean)[1]
    epsilon = K.random_normal(shape = (batch, dimension))
    return latent_mean + K.exp(0.5 * laten_log_value) * epsilon

latent_vector = Lambda(sampling, output_shape = (latent_dimension, ))([latent_mean, latent_log_value])

encoder = Model([encoder_input, condition_input], [latent_mean, latent_log_value, latent_vector])

# encoder.summary()

# Decoder
latent_input = Input(shape = (latent_dimension, ))
latent_concatenate = concatenate([latent_input, condition_input])

latent_layer = Dense(latent_shape[1] * latent_shape[2] * latent_shape[3], activation = 'relu')(latent_concatenate)
latent_layer = Reshape((latent_shape[1], latent_shape[2], latent_shape[3]))(latent_layer)

decoder_layer = Conv2D(16, kernel_size = (3, 3), padding = 'same', activation = 'relu')(latent_layer)
decoder_layer = UpSampling2D(size = (2, 2))(decoder_layer)
decoder_layer = Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu')(decoder_layer)
decoder_layer = UpSampling2D(size = (2, 2))(decoder_layer)
decoder_layer = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(decoder_layer)
decoder_layer = UpSampling2D(size = (2, 2))(decoder_layer)

decoder_output = Conv2D(1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid')(decoder_layer)

decoder = Model([latent_input, condition_input], decoder_output)

# decoder.summary()

output = decoder([encoder([encoder_input, condition_input])[2], condition_input])

# Loss
def loss():
    reconstruction_loss = mse(K.flatten(encoder_input), K.flatten(output))
    reconstruction_loss *= n_pixels * n_pixels
    kl_loss = 1 + latent_log_value - K.square(latent_mean) - K.exp(latent_log_value)
    kl_loss = K.sum(kl_loss, axis = -1)
    kl_loss *= -0.5 * 1.0
    cvae_loss = K.mean(reconstruction_loss + kl_loss)
    keras.losses.cvae_loss = cvae_loss
    return keras.losses.cvae_loss

# Building a model
conditional_vae = Model([encoder_input, condition_input], output)
conditional_vae.add_loss(loss())
conditional_vae.compile(optimizer = 'adam')

# conditional_vae.summary()

conditional_vae.fit([X_train, y_train], batch_size = batch_size, epochs = epochs, validation_data = ([X_validation, y_validation], None), shuffle = True)

n = 10
digit_pixels = n_pixels

figure = np.zeros((digit_pixels * n, digit_pixels * n))

# Linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[ :  : -1]

# digit = np.random.randint(0, n, 1)
# y_label = np.eye(n)[digit]
y_label = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        latent_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict([latent_sample, y_label])
        digit = x_decoded[0].reshape(digit_pixels, digit_pixels)
        figure[i * digit_pixels : (i + 1) * digit_pixels, j * digit_pixels : (j + 1) * digit_pixels] = digit

for i in range(len(X_test)):
    fig = plt.figure(figsize=(10, 10))
    selected = X_test[i]
    selected = selected.reshape(28, 28)
    plot = fig.add_subplot(1, 3, 1)
    plot.set_title('selected')
    plt.axis('off')
    plt.imshow(selected, cmap = 'Greys_r')
  
    real = X_test[((i // 10)*10)]
    real = real.reshape(28,28)
    plot = fig.add_subplot(1, 3, 2)
    plot.set_title('real')
    plt.axis('off')
    plt.imshow(real, cmap = 'Greys_r')

    predictimg = conditional_vae.predict([X_test[i].reshape(1, n_pixels, n_pixels, 1), y_label])
    predictimg = predictimg.reshape(n_pixels, n_pixels)
    plot = fig.add_subplot(1, 3, 3)
    plot.set_title('predict')
    plt.axis('off')
    plt.imshow(predictimg, cmap = 'Greys_r')
    plt.show()