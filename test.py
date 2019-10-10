import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest


model = VGGFace(include_top = False, model = 'resnet50', input_shape = (128, 128, 3))

model.summary()