import cv2
import numpy as np
import time

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    scanned_image = None  # Ensure scanned_image is initialized

    while True:
        # Display the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break  # Exit if the frame isn't captured properly

        # Scan for human
        frame = scan_human(frame)
        
        cv2.imshow("Live Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save the current frame by pressing "s"
        if key == ord('s'): 
            scanned_image = frame.copy()  # Store scanned frame

            saved_image_path = "Screenshot at " + new_name(start_time) + ".jpg"

            cv2.imwrite(saved_image_path, scanned_image)
            print(f"Image saved as {saved_image_path}")

        # Exit by pressing "q"
        elif key == ord('q'):
            break

        # Show the last saved image if one exists
        if scanned_image is not None:
            cv2.imshow("Scan", scanned_image)
    
    # Exit the program
    cap.release()
    cv2.destroyAllWindows()

# Human detector function
def scan_human(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame

# Name generator for new screenshots
def new_name(start_time):
    elapsed_time = round(time.time() - start_time, 3)
    return str(elapsed_time)

if __name__ == "__main__":
    main()
