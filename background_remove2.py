#Import all necessery libraries
import numpy as np
import cv2

#Read the image and perform threshold and get its height and weight
img = cv2.imread('C:/Document/Bitcamp/Project/Frontalization/Imagenius/Data/Korean 224X224X3 filtering/X/001-01-06.jpg.jpg')
h, w = img.shape[:2]

# Transform to gray colorspace and blur the image.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

# Make a fake rectangle arround the image that will seperate the main contour.
cv2.rectangle(blur, (0,0), (w,h), (255,255,255), 10)

# Perform Otsu threshold.
_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Create a mask for bitwise operation
mask = np.zeros((h, w), np.uint8)

# Search for contours and iterate over contours. Make threshold for size to
# eliminate others.
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for i in contours:
    cnt = cv2.contourArea(i)
    if 1000000 >cnt > 100000:
        cv2.drawContours(mask, [i],-1, 255, -1)


# Perform the bitwise operation.
res = cv2.bitwise_and(img, img, mask=mask)

# Display the result.
cv2.imwrite('mole_res.jpg', res)
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()