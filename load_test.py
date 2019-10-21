from datagenerator_predict import DataGenerator
from glob import glob
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

time = 2

batch_size = 128
save_path = 'D:/Generated Image/Testing' + str(time) + '/'
number = 1

X = glob('D:/Bitcamp/Project/Frontalization/Imagenius/Data/Test image/*jpg')

datagenerator = DataGenerator(X, batch_size = batch_size)

generator = load_model('D:/generator_epoch_30.h5')

for i in range(datagenerator.__len__()):
    side_image = datagenerator.__getitem__(i)
    
    generated_image = 0.5 * generator.predict(side_image) + 0.5

    side_image = (127.5 * (side_image + 1)).astype(np.uint8)

    for j in range(batch_size):
        plt.figure()

        for k in range(1):
            original_side_face_image_plot = plt.subplot(1, 2, 1)
            original_side_face_image_plot.set_title('Origninal side image')
            original_side_face_image_plot.imshow(side_image[j])

            generated_image_plot = plt.subplot(1, 2, 2)
            generated_image_plot.set_title('Generated image (front image)')
            generated_image_plot.imshow(generated_image[j])  

            generated_image_plot.axis('off')
            original_side_face_image_plot.axis('off')

            # plt.show()
            
        # Check folder presence
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        save_name = 'Testing%d.png' % (number)
        save_name = os.path.join(save_path, save_name)

        plt.savefig(save_name)
        plt.close()

        number += 1