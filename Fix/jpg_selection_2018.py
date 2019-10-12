from glob import glob
import os.path
import cv2

save_path = "d:/Selected_Img"
people = glob("d:/AIhub/2018/High_Resolution_2018/High_Resolution_2018/All/*")
n = 100
# print(people)
for person in people:
    n += 1
    # print(person)
    person_num = person[-8:]
    # print(person_num)
    for l in range(6):
        personpath = []
        personpath = (person[:-9] + '/'+ person_num +'/S001/L0' + str(l+1) + '/E01/*.jpg')
        # print(personpath)
        imgs = glob(personpath)
        # print(imgs)
        for img in imgs:
            C_index = img.find('C')
            angle = int(img[C_index+1:-4])
            # print(angle)
            if angle > 13:
                continue
            # print(img)
            path = save_path + '/{0:03}-{1}-{2:02}.jpg'.format(n, l+1, angle)
            # print(path)
            image = cv2.imread(img)
            cv2.imwrite(path, image)

print(n)
