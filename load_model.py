import keras
import cv2
from keras.models import load_model
from skimage import data, io
import Utils_model as Utils_model
from Utils_model import VGG_LOSS
import numpy as np
import matplotlib.pyplot as plt
import os
image_shape = (120, 120, 3)
loss = VGG_LOSS(image_shape)
optimizer = Utils_model.get_optimizer()

path = os.getcwd()
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def gen(file_name):

    image = data.imread(path+"/upload/%s"%(file_name))
    cv2.imwrite(path + "/gen_output/before1.png", image)
    
    plt.show()
    image = cv2.resize(image, (image_shape[0], image_shape[1]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path+"/gen_output/before2.png", image)
    image = image.reshape((1,image_shape[0], image_shape[1],image_shape[2]))
    print(image.shape)
    image = normalize(image)
    model = load_model(path+'/gen/gen_model100.h5',custom_objects={'vgg_loss': loss.vgg_loss})

    
    pre_img=model.predict(image)
    print(pre_img)
    pre_img = denormalize(pre_img)
    print(pre_img.shape)
    pre_img = pre_img.reshape((image_shape[0], image_shape[1],image_shape[2]))
    b, g, r = cv2.split(pre_img)
    pre_img = cv2.merge([r,g,b])
    cv2.imwrite(path+"/gen_output/%s"%(file_name), pre_img)

    
