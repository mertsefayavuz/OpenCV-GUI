import cv2
import numpy as np

image = cv2.imread('testimg.png')

# Define the transformation matrix (translation matrix).
M = np.float32([[0.5, 1, 100], [0.2, 1, -100]])

# Apply the affine transformation
affine_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

cv2.imshow('Original Image', image)
cv2.imshow('Affine Transformed Image', affine_image)

cv2.imwrite('testimg affined.png', affine_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 3D Translation:
# [1, 0.5, 0, dx], [0.75, 1, 0, dy], [0, 0, 1, dz]
#       |              |              > for every change in z dimension, none of other dimension changes
#       |              > for every change in y dimension, x dimension changes by 0.75
#       > for every change in x dimension, y changes by 0.5
# The value of the changes might be positive or negative, which decides the direction of movement 
# Watch this short video for better comprehension: https://www.youtube.com/watch?v=AheaTd_l5Is

