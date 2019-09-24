import glob
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

size = 64
num = 14
images = glob.glob('D:/Bitcamp/Project/Frontalization/Data/origin_image_sample/*.jpg')
x = []
y = []

for fname in images:
    im = pilimg.open(fname)
    # print(fname)
    im = im.resize((size,size))
    # im.show()
    # print(im.size)
    if (fname % num == 10)|(fname % num == 11)|(fname % num == 12)|(fname % num == 13):
        continue
    else :
        img = np.array(im)
        lable = np.array(images[(fname // num) + 1] * 10)
    # print(img.shape)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    # print(img.shape)
    x.append(img)
    y.append(lable)

x = np.array(x)
print(x.shape)
np.save("D:/{}_{}.npy".format(size, num), x)
y = np.array(y)
print(y.shape)
print(y)
np.save("D:/{}_{}.npy".format(size, num), y)