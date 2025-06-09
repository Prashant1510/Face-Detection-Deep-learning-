from ultralytics import YOLO

# Load YOLOv8 model (small, fast version)
model = YOLO('yolov8n.pt')  # Replace with 'yolov8s.pt' for better accuracy

# Train the model using your dataset
results = model.train(
    data=r'C:\Users\Lenovo\Desktop\object detection\face_detection_yolov8\data.yaml',  # Path to your data.yaml file
    epochs=50,         # Number of training epochs (you can increase this for better accuracy)
    imgsz=200,         # Image size for training (500x500 in this case, adjust as needed)
    batch=50,          # Batch size (adjust depending on your system's memory, try lowering it if you face memory issues)
    name='face-detection-yolov8',  # Custom name for the training run
    workers=2          # Number of data-loading workers (adjust based on your CPU, 2-4 should be fine for your system)
)

# Save the trained model to a specific path
model.save(r'C:\Users\Lenovo\Desktop\object detection\face_detection_yolov8\models/face_detection_model.pt')

# Print results summary
print("Training completed. Results summary:")
print(results)

