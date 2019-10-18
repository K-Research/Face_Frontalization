from keras.models import load_model, Model, Input, model_from_json
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy
from keras.preprocessing.image import load_img, img_to_array
from datagenerator_predict import DataGenerator
from glob import glob
import os

time = 1

save_path = 'D:/Generated Image/Testing' + str(time) + '/'
number = 1

X = glob('D:/TEST/*jpg')

datagenerator = DataGenerator(X, batch_size = 64)

generator = load_model('D:/generator_epoch_10.h5')

for i in range(datagenerator.__len__()):
    image = datagenerator.__getitem__(i)
    
    original_image = ((image + 1) * 127.5).astype(numpy.uint8)

    generated_image = 0.5 * generator.predict(image) + 0.5

    original_side_face_image_plot = plt.subplot(1, 2, 1)
    original_side_face_image_plot.set_title('Origninal front image')
    original_side_face_image_plot.imshow(original_image[i])

    generated_image_plot = plt.subplot(1, 2, 2)
    generated_image_plot.set_title('Generated image (front image)')
    generated_image_plot.imshow(generated_image[i])  

    generated_image_plot.axis('off')
    original_side_face_image_plot.axis('off')

    number += 1

    # plt.show()
    
    # Check folder presence
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    save_name = 'Testing%d.png' % (number)
    save_name = os.path.join(save_path, save_name)

    plt.savefig(save_name)
    plt.close()