import cv2
import sys

# Reading the image in gray scale
image = cv2.imread('testimg.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur and Canny
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow('Original', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()