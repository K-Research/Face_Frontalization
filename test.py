import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest


model = VGGFace(include_top = True, model = 'senet50')

model.summary()