import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest

model1 = VGGFace(model = 'vgg16')
model2 = VGGFace(model = 'resnet50')
model3 = VGGFace(model = 'senet50')