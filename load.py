from keras.models import load_model, Model, Input, model_from_json
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import keras.backend as K
# AI Model
class VGG_LOSS(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape
    def vgg19_loss(self, true, prediction):
        vgg19 = VGG19(include_top = False, weights = 'imagenet', input_shape = (self.image_shape))
        # Make trainable as False

        vgg19.trainable = False

        for layer in vgg19.layers:
            layer.trainable = False
        
        model = Model(inputs = vgg19.input, outputs = vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(true) - model(prediction)))

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
loss = VGG_LOSS(image_shape=(128, 128, 3))
image = cv2.imread('G:/001-2-09.jpg')

print(image.shape)

image = img_to_array(image)

print(image.shape)

images = []
images.append(image)
images.append(image)

images = numpy.array(images)

print(images.shape)

images = (images.astype(numpy.float32) - 127.5) / 127.5

model = load_model('G:/generator_epoch_15.h5', custom_objects = {'vgg19_loss' : loss.vgg19_loss})

model.summary()

generated_image = model.predict(images)

pred = (generated_image +1) * 127.5

pred = pred.astype(numpy.uint8)

print(generated_image)

b, g, r = cv2.split(pred[0])

pred_image = cv2.merge([r, g, b])

plt.imshow(pred_image)
plt.show()