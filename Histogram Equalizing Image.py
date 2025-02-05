import cv2

image = cv2.imread('testimg.png')

image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization 
equalized_image = cv2.equalizeHist(image_GRAY) 

cv2.imshow('Original Image', image)
cv2.imshow('Gray Image', image_GRAY)
cv2.imshow('Equalized Image', equalized_image) 

cv2.imwrite('testimg equalized.png', equalized_image)

cv2.waitKey(0) 
cv2.destroyAllWindows()