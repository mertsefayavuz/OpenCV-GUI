import cv2 
import numpy as np 

image = cv2.imread('testimg.png') 
cv2.imshow('Original Image', image) 

operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Modifying the data type to set to 32-bit floating point 
operatedImage = np.float32(operatedImage) 

# Apply the Harris corner detection method 
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 

# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 

# Reverting back to the original image, with optimal threshold value 
image[dest > 0.01 * dest.max()]=[0, 0, 255] 

cv2.imshow('Image with Borders', image) 

cv2.waitKey(0)
cv2.destroyAllWindows() 
