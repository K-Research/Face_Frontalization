from keras.models import load_model, Model
from datagenerator_read_dir_face import DataGenerator
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

model = load_model("D:/trainedmodel/batch32/20_51.h5")
batch_size = 32
save_interval = 1
test = glob("D:/TEST/*jpg")
n_show_image = 1
line = 8


class prediction():
    def __init__(self):
        self.model = model
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.DG = DataGenerator(test, batch_size = batch_size)
        self.line = line

    def predicted(self, batch_size, save_interval):
        fake = np.zeros((batch_size))
        real = np.ones((batch_size))

        for j in range(self.DG.__len__()):
            side_images = self.DG.__getitem__(j)

            generated_images = self.model.predict(side_images)

            if j % save_interval == 0:
                save_path = 'D:/Generated Image/predict' + str(line) + '/'
                self.save_image(batch = j, side_image = side_images, save_path = save_path)

    def save_image(self, batch, side_image, save_path):
        generated_image = 0.5 * self.model.predict(side_image) + 0.5
     
        side_image = (255 * ((side_image)+1)/2).astype(np.uint8)
        
        for i in range(self.batch_size):
            plt.figure(figsize = (8, 2))

            # Adjust the interval of the image
            plt.subplots_adjust(wspace = 0.6)

            # Show image (first column : original side image, second column : original front image, third column = generated image(front image))
            for m in range(n_show_image):
                generated_image_plot = plt.subplot(1, 2, 1)
                generated_image_plot.set_title('Generated image (front image)')
                plt.imshow(generated_image[i])

                original_side_face_image_plot = plt.subplot(1, 2, 2)
                original_side_face_image_plot.set_title('Origninal side image')
                plt.imshow(side_image[i])

                # Don't show axis of x and y
                generated_image_plot.axis('off')
                original_side_face_image_plot.axis('off')

                # plt.show()

            save_path = save_path

            # Check folder presence
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_name = '%d-%d.png' % (batch, i)
            save_name = os.path.join(save_path, save_name)
        
            plt.savefig(save_name)
            plt.close()


import numpy as np
from skimage.io import imread
import keras
from glob import glob
import PIL.Image as pilimg

class DataGenerator(keras.utils.Sequence):
    def __init__(self, sideslist, batch_size = 32, dim = (224, 224), n_channels = 3, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size
        self.sideslist = sideslist
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sideslist) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        sideslist_temp = [self.sideslist[k] for k in indexes]

        sides = self.__data_generation(sideslist_temp)

        return sides

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sideslist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sideslist):
        X = np.empty((self.batch_size, * self.dim, self.n_channels))
        i = 0

        for sidename in sideslist:
            # print("sidename : " + sidename)
            side = pilimg.open(sidename)
            side = np.array(side)
            # pilimg._show(side)
            # side.close()
            X[i] = side
            # print(X.shape)
            i += 1

        return self.preprossing(X)
    
    def preprossing(self, img):
        return (img / 127.5 - 1)

if __name__ == '__main__':
    prediction = prediction()
    prediction.predicted(batch_size = batch_size, save_interval = n_show_image)