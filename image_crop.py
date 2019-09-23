import cv2
import os

def facecrop(image_path):
    cascade = cv2.CascadeClassifier('D:/Bitcamp/Project/Frontalization/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')

    image = cv2.imread(image_path)

    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for i in faces:
        x, y, w, h = [v for v in i]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255))

        find_face = image[y : y + h, x : x + w]
        find_face = cv2.resize(find_face, (28, 28))

        directory, file_name = os.path.split(image_path)
        cv2.imwrite(save_path + file_name, find_face)

    return

image_path = 'D:/Download/4.jpg'
save_path = 'D:/cropped_image/'

if __name__ == '__main__':
    facecrop(image_path)