import numpy as np
import cv2
 
image = cv2.imread('testimg.png')
compare_image = cv2.imread('testimg resized.png')
 
image_GRAY = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
compare_image_GRAY = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
 
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
 
# Now detect the keypoints and compute the descriptors for the images
image_keypoints, image_descriptors = orb.detectAndCompute(image_GRAY,None)
compare_image_keypoints, compare_image_descriptors = orb.detectAndCompute(compare_image_GRAY,None)

# Initialize the Matcher for matching the keypoints and then match the keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(image_descriptors,compare_image_descriptors)
 
# Draw the matches to the final image
final_image = cv2.drawMatches(image_GRAY, image_keypoints, compare_image, compare_image_keypoints, matches[:20],None)
 
final_image = cv2.resize(final_image, (1000,650))

cv2.imshow("Matches", final_image)

cv2.imwrite('testimg matched.png', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()