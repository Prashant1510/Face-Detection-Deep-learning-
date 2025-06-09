# Convert the YOLOv8 model to TensorFlow Lite
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO(r'C:\Users\Lenovo\Desktop\object detection\runs\detect\face-detection-yolov8\weights\best.pt')

# Export to TensorFlow Lite format
model.export(format='tflite', dynamic=False)  # Set 'dynamic' to False for optimization
