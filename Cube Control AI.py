import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Cube parameters
scale_factor = 15
cube_vertices = scale_factor * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32)

cube_edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]

# Cube faces with colors (BGR format)
cube_faces = [
    ([0, 1, 2, 3], (0, 0, 255)),     # Front (Red)
    ([4, 5, 6, 7], (0, 255, 0)),     # Back (Green)
    ([0, 3, 7, 4], (255, 0, 0)),     # Left (Blue)
    ([1, 5, 6, 2], (0, 255, 255)),   # Right (Yellow)
    ([2, 3, 7, 6], (255, 0, 255)),   # Top (Magenta)
    ([0, 4, 5, 1], (255, 255, 0))    # Bottom (Cyan)
]

# Camera parameters
focal_length = 500
cube_window_size = 600
principal_point = (cube_window_size//2, cube_window_size//2)

# Cube control parameters
current_rotation_x = 0.0
current_rotation_y = 0.0
target_rotation_x = 0.0
target_rotation_y = 0.0
current_zoom = 3.0
target_zoom = 3.0
smoothing_factor = 0.1
rotation_sensitivity = 3  # Increased sensitivity for rotation

def project_3d_to_2d(points, rotation_x, rotation_y, zoom):
    # Rotation matrices
    rx = np.array([[1, 0, 0],
                  [0, math.cos(rotation_x), -math.sin(rotation_x)],
                  [0, math.sin(rotation_x), math.cos(rotation_x)]])
    
    ry = np.array([[math.cos(rotation_y), 0, math.sin(rotation_y)],
                  [0, 1, 0],
                  [-math.sin(rotation_y), 0, math.cos(rotation_y)]])
    
    # Combine rotations
    rotation_matrix = rx @ ry
    
    # Apply transformations
    transformed = points @ rotation_matrix.T
    transformed *= zoom
    
    # Project to 2D
    projected = []
    for point in transformed:
        z = point[2] + focal_length
        x = point[0] * focal_length / z + principal_point[0]
        y = point[1] * focal_length / z + principal_point[1]
        projected.append((int(x), int(y)))
    
    return projected

cap = cv2.VideoCapture(0)
cv2.namedWindow('3D Cube', cv2.WINDOW_NORMAL)
cv2.resizeWindow('3D Cube', cube_window_size, cube_window_size)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Process hands
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    left_hand_pos = None
    right_hand_distance = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label
            
            if hand_type == 'Left':
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                left_hand_pos = (wrist.x, wrist.y)
            elif hand_type == 'Right':
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                right_hand_distance = math.hypot(thumb.x - index.x, thumb.y - index.y)
    
    # Update targets
    if left_hand_pos:
        # Increased sensitivity with direct mapping to rotation angles
        target_rotation_x = (left_hand_pos[1] - 0.5) * 3 * rotation_sensitivity  # Increased multiplier
        target_rotation_y = (left_hand_pos[0] - 0.5) * 3 * rotation_sensitivity

    if right_hand_distance:
        # Reversed zoom interpolation
        target_zoom = np.interp(right_hand_distance, [0.02, 0.2], [0.5, 5.0])  # Swapped output range
    
    # Smooth transitions
    current_rotation_x += (target_rotation_x - current_rotation_x) * smoothing_factor
    current_rotation_y += (target_rotation_y - current_rotation_y) * smoothing_factor
    current_zoom += (target_zoom - current_zoom) * smoothing_factor
    
    # Create cube visualization
    cube_image = np.zeros((cube_window_size, cube_window_size, 3), dtype=np.uint8)
    projected = project_3d_to_2d(cube_vertices, current_rotation_x, current_rotation_y, current_zoom)
    
    # Draw faces
    for face_indices, color in cube_faces:
        pts = np.array([projected[i] for i in face_indices], dtype=np.int32)
        cv2.fillPoly(cube_image, [pts], color)
    
    # Draw edges
    for edge in cube_edges:
        start = projected[edge[0]]
        end = projected[edge[1]]
        cv2.line(cube_image, start, end, (255, 255, 255), 2)
    
    # Show windows
    cv2.imshow('Hand Tracking', frame)
    cv2.imshow('3D Cube', cube_image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()