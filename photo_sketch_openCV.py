import cv2
import numpy as np


image = cv2.imread("/home/lina/Desktop/horse.jpeg")

# step 1: convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

cv2.imshow('gray_image',gray_image )
cv2.waitKey(0)

# step 2: blur the image

blurred_image = cv2.GaussianBlur( gray_image , (3,3), 10 ) 

cv2.imshow('blurred_image',blurred_image )
cv2.waitKey(0)


# step 3: divide the gray image by the blurred image

sketched_image = np.divide(gray_image , blurred_image)

cv2.imshow('sketched_image',sketched_image )
cv2.waitKey(0)



