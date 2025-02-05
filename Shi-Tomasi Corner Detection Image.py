import cv2
import numpy as np
import matplotlib.pyplot as plt  

img = cv2.imread('testimg.png')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners using Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10)
corners = np.int32(corners)

# Draw circles on the corners
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (255, 0, 0), -1)  # Red circles (BGR format)

# Convert BGR image to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()  # Use plt.show() instead of "%matplotlib inline"