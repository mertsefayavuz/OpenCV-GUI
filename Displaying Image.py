import cv2 as cv

# Reading the image
image = cv.imread(cv.samples.findFile("testimg.png"))

# Displaying the image
cv.imshow("testimg.png", image)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()