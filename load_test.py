from keras.models import load_model, Model, Input, model_from_json
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing.image import load_img, img_to_array
# AI Model

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

image = cv2.imread('C:/Document/Bitcamp/001-2-09.jpg')

print(image.shape)

image = img_to_array(image)

print(image.shape)

images = []
images.append(image)
images.append(image)

images = numpy.array(images)

print(images.shape)

json_file=open("C:/Document/Bitcamp/generator_model.json","r")
loaded_json=json_file.read()
json_file.close()

model=model_from_json(loaded_json)
model.load_weights("C:/Document/Bitcamp/generator_weights_epoch_10.h5")
MODEL=model
# MODEL.summary()
generated_image = 0.5 * MODEL.predict(images) + 0.5

plt.imshow(generated_image[0])
plt.show()