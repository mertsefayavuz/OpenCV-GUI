import cv2

image = cv2.imread('testimg.png')

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

cv2.imshow('Keypoints', image_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()