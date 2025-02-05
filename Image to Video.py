import cv2
import os

# Use raw string to handle special characters
folder_path = r"C:\OpenCV Guide\Detected Motions"

# List all files and folders
files = os.listdir(folder_path)
for file in files:
    print(file)

directory = input("Choose an archive name from above: ")

video_path = os.path.join(folder_path, directory)  # Ensure proper path joining

# Debug: Check if the selected directory exists
if not os.path.exists(video_path):
    print("Error: The selected directory does not exist!")
    exit()

# Get all image files sorted alphabetically
images = sorted([f for f in os.listdir(video_path) if f.endswith((".png", ".jpg", ".jpeg"))])

# Debug: Check if images were found
if not images:
    print("Error: No images found in the selected folder!")
    exit()

# Ensure proper file path joining
first_image_path = os.path.join(video_path, images[0])
print("First image path:", first_image_path)

# Read the first image
first_image = cv2.imread(first_image_path)

# Debug: Check if the first image loaded correctly
if first_image is None:
    print("Error: Failed to load the first image! Check file path or file integrity.")
    exit()

height, width, layers = first_image.shape

# Display images as a video
for image in images:
    img_path = os.path.join(video_path, image)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Warning: Skipping unreadable image: {image}")
        continue  # Skip unreadable images

    cv2.imshow("Image Slideshow", frame)

    # Wait for 150ms (0.15 second) per frame, press 'q' to exit
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
