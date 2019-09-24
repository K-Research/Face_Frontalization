import glob
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

size = 64
num = 14
images = glob.glob('D:/Bitcamp/Project/Frontalization/Data/origin_image_sample/*.jpg')
x = []
y = []
n = 0
for fname in images:
    im = pilimg.open(fname)
    # print(fname)
    im = im.resize((size,size))
    # im.show()
    # print(im.size)
    if (n % num == 10) or (n % num == 11) or (n % num == 12) or (n % num == 13):
        n += 1
        continue
    else :
        img = np.array(im)
        img_y =  pilimg.open(images[((n // num) + 1 )* 10])
        img_y = img_y.resize((size,size))
        lable = np.array(img_y)
    # print(img.shape)
    # img = img.reshape(img.shape[0], img.shape[1], 1)
    # print(img.shape)
        x.append(img)
        y.append(lable)
    n += 1

x = np.array(x)
print(x.shape)
np.save("D:/X_{}_{}.npy".format(size, num), x)
y = np.array(y)
print(y.shape)
np.save("D:/Y_{}_{}.npy".format(size, num), y)