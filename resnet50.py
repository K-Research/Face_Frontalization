import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest
from keras.layers import Flatten
from keras.models import Model

model = VGGFace(include_top = True, model = 'senet50', pooling = 'max')
# vgg16_layer = VGGFace(include_top = False, model = 'resnet50', weights = 'vggface', input_shape = (128, 128, 3))
# last_layer = Flatten()

# model.get_layer.out

# model = Model(vgg16_layer, last_layer)

model.summary()