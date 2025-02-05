import cv2
import numpy as np
import time
import os
import shutil

def main():
    # Create directory for detected motions
    directory = input("Input a name for directory of recorded motions: ")
    save_path = os.path.join(r"C:\OpenCV Guide\Detected Motions", directory)  # Change to a safe directory
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    
    flag = 0
    start_time = time.time()
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    while True:
        ret, frame = cap.read()

        # Reset flag every 0.05 seconds
        if timer(start_time) % 0.05 < 0.01:
            flag = 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Detect motion: Calculate the difference between the background and the current frame
        frame_diff = cv2.absdiff(gray_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 5, 255, cv2.THRESH_BINARY)

        # Find contours of the detected motion
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for significant motion regions
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the motion mask (thresh) has white areas (motion detected)
        if np.sum(thresh) > 0 and flag == 1:  # If there is motionr
            flag = 0

            # Save the image with the rectangle drawn around the motion
            if timer(start_time) < 10:
                elapsed_time = "00" + str(timer(start_time))
            elif timer(start_time) < 100:
                elapsed_time = "0" + str(timer(start_time))
            else:
                elapsed_time = str(timer(start_time))
            saved_image_path = os.path.join(save_path, f"Detected motion at {elapsed_time}.png")
            cv2.imwrite(saved_image_path, frame)
            print(f"Saved image: {saved_image_path}")

        # Display live camera feed and the motion mask
        cv2.imshow("Live Camera", frame)
        cv2.imshow("Detected Motions", thresh)

        # Update the background with the current frame
        gray_frame = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Move saved images to the correct directory (if they are not there already)
    source_dir = r'C:\OpenCV Guide'
    target_dir = save_path

    # Move the saved images to the specified directory
    for image_name in os.listdir(source_dir):
        if image_name.startswith('Detected motion at '):  # 'Detected motion at ' prefix
            source_path = os.path.join(source_dir, image_name)
            target_path = os.path.join(target_dir, image_name)
            shutil.move(source_path, target_path)

# Function to evaluate the time passed till last call
def timer(start_time):
    passed_time = round(time.time() - start_time, 3)
    return passed_time

if __name__ == "__main__":
    main()
