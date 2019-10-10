import numpy as np
from skimage.io import imread
import keras
from glob import glob
import PIL.Image as pilimg

class DataGenerator(keras.utils.Sequence):
    def __init__(self, imgslist, batch_size=32, dim = (128, 128), n_channels = 3, n_classes = 13, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = []
        self.imgslist = imgslist
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.imgslist) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        imgslist_temp = [self.imgslist[k] for k in indexes]

        imgs, labels = self.__data_generation(imgslist_temp)

        return imgs, labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgslist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, imgslist):
        X = np.empty((self.batch_size, * self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, imgname in enumerate(imgslist):
            # print(imgname)
            temp = imgname.find('-')
            temp = imgname[temp+1:temp+3]
            if temp[0] == '0':
                temp = temp[1]
            label = int(temp)-1
            # print(label)
            img = pilimg.open(imgname)
            # pilimg._show(img)
            X[i] = img
            y[i] = label
            print(X.shape)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':

    imgslist = glob("D:/i/*jpg")

    gd = DataGenerator(imgslist)