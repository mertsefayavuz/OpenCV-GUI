from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pretrained for humans)
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread('testimg.png')

# Run detection
results = model(image)

# Draw bounding boxes for "person" class (class ID 0)
for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show result
cv2.imshow("YOLOv8 Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
