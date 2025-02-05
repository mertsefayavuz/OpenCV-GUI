import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('testimg.png', cv2.IMREAD_GRAYSCALE)

# Calculate HOG features
features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Rescale HOG features for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the original image and HOG features
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.title('HOG Features')
plt.axis('off')
plt.show()
