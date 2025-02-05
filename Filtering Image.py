import cv2
import numpy as np

image = cv2.imread("testimg.png")

# Box Filter (Averaging)
kernel_size = (5, 5)
image_box_filtered = cv2.blur(image, kernel_size)

# Gaussian Blur
image_gaussian_filtered = cv2.GaussianBlur(image, kernel_size, 0)

# Median Blur
image_median_filtered = cv2.medianBlur(image, 5)

# Bilateral Filtering
image_bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)

# Edge Detection (Sobel)
image_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
image_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
image_sobel_combined = cv2.magnitude(image_sobel_x, image_sobel_y)

# Edge Detection (Laplacian)
image_laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)

# Emboss Filter
emboss_kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
image_emboss_filtered = cv2.filter2D(image, -1, emboss_kernel)

# Sharpening Filter
sharpen_kernel = np.array([[0, -1,  0],
                            [-1, 5, -1],
                            [0, -1,  0]])
image_sharpened = cv2.filter2D(image, -1, sharpen_kernel)

cv2.imshow("Original Image", image)
cv2.imshow("Box Filtered", image_box_filtered)
cv2.imshow("Gaussian Blur", image_gaussian_filtered)
cv2.imshow("Median Blur", image_median_filtered)
cv2.imshow("Bilateral Filter", image_bilateral_filtered)
cv2.imshow("Sobel Combined", image_sobel_combined.astype(np.uint8))
cv2.imshow("Laplacian Filter", image_laplacian_filtered.astype(np.uint8))
cv2.imshow("Emboss Filter", image_emboss_filtered)
cv2.imshow("Sharpened", image_sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
