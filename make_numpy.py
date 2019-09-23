import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pilimg

shape = 28

def x2numpy(directory, name):
    images = glob.glob(directory + '/*.jpg')
    np_image = []
    for fname in images:
        im = pilimg.open(fname)
        im = im.resize((shape,shape))
        img = np.array(im)
        img = img.reshape(img.shape[0], img.shape[1], 1)
        np_image.append(img)

    np_image = np.array(np_image)
    print('x shape: ', np_image.shape)
    np.save('D:/' + name + ".npy", np_image)
    print('saved')

    
def y2numpy(directory, name):
    images = glob.glob(directory + '/*.jpg')
    np_image = []
    for fname in images:
        im = pilimg.open(fname)
        im = im.resize((shape,shape))
        img = np.array(im)
        img = img.reshape(img.shape[0], img.shape[1], 1)

        for i in range(10):
            np_image.append(img)
        

    np_image = np.array(np_image)
    print('y shape: ', np_image.shape) 
    np.save('D:/' + name + ".npy", np_image)
    print('saved')



def image2numpy(directory):
    im = pilimg.open('./lsm3.jpg')
    im = im.resize((shape,shape)).convert('L')
    img = np.array(im)
    print(img.shape)
    img = img.reshape(img.shape[0], img.shape[1], 1)

    print('x shape: ', img.shape)
    np.save("lsm64.npy", img)
    print('saved')



x2numpy('D:/Bitcamp/Project/Frontalization/Numpy/X_10_per_person/', 'X_10_per_person')
y2numpy('D:/Bitcamp/Project/Frontalization/Numpy/Y_10_per_person/', 'Y_10_per_person')

# image2numpy('aa')

# numpy_image('x_test')
# numpy_image('y_test')