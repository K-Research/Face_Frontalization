from glob import glob
import cv2
from crop_img import crop_progress
import numpy as np

path = "d:/Selected_Img"
save_Xpath = "d:/X/"
save_Ypath = "d:/Y/"
imgs = glob(path+"/*.jpg")
# crop_front_img = np.zeros(300)
# print(imgs)
# for front in range(1, 301):
#     front = "{0:03}".format(front)
#     # print(front)
#     front_img_name = path + "/" + front + "-2-07.jpg"
#     print(front_img_name)
#     front_image = cv2.imread(front_img_name, cv2.IMREAD_COLOR)
#     crop_front_img = crop_progress(front_image)
#     # print(crop_front_img.shape)
#     # cv2.imshow("crop_img",crop_front_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

for img in imgs:
    temp = img.find("\\")
    # print(temp)
    img_name = img[temp+1:]
    # print(img_name)
    person, l, c = img_name.split('.')[0].split('-')
    front_img_name = path + "/" + person + "-2-07.jpg"
    # print(front_img_name)
    front_image = cv2.imread(front_img_name, cv2.IMREAD_COLOR)
    crop_front_img = crop_progress(front_image)
    # print(person, l, c)
    # img = img[:temp] +'/'+ img_name
    # print(img)
    if c == '07':
        continue
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    crop_img = crop_progress(image)
    if crop_img is None:
        continue

    cv2.imwrite(save_Xpath + img_name + ".jpg", crop_img)
    cv2.imwrite(save_Ypath + img_name + ".jpg", crop_front_img)
    # print(ljh_hansome))
    # print(crop_img.shape)
    # cv2.imshow("crop_img",crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()