import cv2

image = cv2.imread('testimg.png')

# Convert BGR to RGB
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow('Original Image', image)
cv2.imshow('RGB Image', image_RGB)

cv2.imwrite('testimg recoloured.png', image_RGB)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Conversion codes are written as "source to target" ("source", "2", "target") which are: 
#BGR to RGB:         cv2.COLOR_BGR2RGB
#RGB to BGR:         cv2.COLOR_RGB2BGR
#BGR to Grayscale:   cv2.COLOR_BGR2GRAY
#Grayscale to BGR:   cv2.COLOR_GRAY2BGR
#BGR to HSV:         cv2.COLOR_BGR2HSV
#HSV to BGR:         cv2.COLOR_HSV2BGR
#RGB to HSV:         cv2.COLOR_RGB2HSV
#HSV to RGB:         cv2.COLOR_HSV2RGB