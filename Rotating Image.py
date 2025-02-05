import cv2

# Load the image
image = cv2.imread(cv2.samples.findFile("testimg.png"))

# Get the image dimensions
(height, width) = image.shape[:2]

# Define the rotation parameters
angle = 45  # Rotation angle in degrees
center = (width // 2, height // 2)  # Center of rotation

# Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# Perform the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow("Original Image", image)
cv2.imshow("Rotated Image", rotated_image)

cv2.imwrite('testimg rotated.png', rotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()