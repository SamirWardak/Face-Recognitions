import cv2
import numpy as np

image = cv2.imread('temp_store/Samir.jpg')
# I just resized the image to a quarter of its original size
image1 = cv2.resize(image, (0, 0), None, .25, .25)
image2 = cv2.resize(image, (0, 0), None, .25, .25)

# Make the grey scale image have three channels

numpy_horizontal = np.hstack((image1, image2))
cv2.imshow('Numpy Horizontal', numpy_horizontal)

cv2.waitKey()