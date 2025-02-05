import cv2

image = cv2.imread(cv2.samples.findFile("testimg.png"))

# Defining new sizes
new_widht = 400
new_height = 200

# Resizing the image as "resized_image"
resized_image = cv2.resize(image, (new_widht, new_height))

cv2.imshow("Original Image", image)
cv2.imshow('Resized Image', image)

# Also if you want to save the changes as a new image, you can declare a spesific condition, for example it is "when "s" is pressed" here:
save_key = cv2.waitKey(0)
if save_key == ord("s"):
    cv2.imwrite('testimg resized.png', resized_image)
    re_show_key = cv2.waitKey(0)
    if save_key == ord("d"):
        cv2.imshow('New image', )