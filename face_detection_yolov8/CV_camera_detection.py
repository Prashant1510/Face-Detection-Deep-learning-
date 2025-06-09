import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model (use the path to your best.pt file)
model = YOLO('/home/prashant/My Drive/AI Projects/Pandas/object detection/runs/detect/face-detection-yolov8/weights/best.pt')

# Open the webcam (use 0 for default webcam, or 1, 2 if you have multiple cameras)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame (directly in BGR format)
    results = model.predict(source=frame, show=False)

    # Extract the detections
    detections = results[0].boxes  # Extract detection results for the first frame
    
    # Check if any detections were made
    if len(detections) > 0:
        for box in detections:
            # Get box coordinates and confidence score
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Class ID (should be 'face')

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label with confidence
            label = f"Face {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print("No faces detected")

    # Display the frame with detections
    cv2.imshow('YOLOv8 Face Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
