import glob
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

size = 64

images = glob.glob('D:/FaceData/origin_image_sample/*.jpg')
x = []
y = []
y_ohe = []
dir_name = 'origin_image_sample'
color_cl = 'c'
area = (150, 80, 500, 400)

for fname in images:
    im = pilimg.open(fname)
    # print(fname)
    # im = im.resize((size,size))
    # im.show()
    cropped = im.crop(area)
    cropped = cropped.resize((size, size))
    # cropped.show()

    # print(im.size)
    # img = np.array(im)
    # # print(img.shape)
    # # img = img.reshape(img.shape[0], img.shape[1], 1)
    # print(img.shape)
    # x.append(img)
    
    print(cropped.size)
    img = np.array(cropped)
    # print(img.shape)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    print(img.shape)
    x.append(img)

x = np.array(x)
print(x.shape)

np.save(("./swh/npy/cvaex_{}_{}_{}_crop.npy".format(dir_name, str(size), color_cl)), x)

for i in range(x.shape[0]):
    temp = i % 14
    y.append(temp)

y = np.array(y)
print(y.shape)
print(y)
np.save(("./swh/npy/cvaey_{}_{}_{}_crop.npy".format(dir_name, str(size), color_cl)), y)