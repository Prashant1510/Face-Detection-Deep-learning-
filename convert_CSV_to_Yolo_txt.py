import os
import pandas as pd
from PIL import Image  # Used to get the image size

# Set the paths
image_folder = r"/home/prashant/My Drive/AI Projects/Pandas/object detection/face_detection_yolov8/dataset/images"
label_folder = r"/home/prashant/My Drive/AI Projects/Pandas/object detection/face_detection_yolov8/dataset/labels"
csv_file = r"/home/prashant/My Drive/AI Projects/Pandas/object detection/face_detection_yolov8/dataset/faces.csv"

# Create the label folder if it doesn't exist
os.makedirs(label_folder, exist_ok=True)

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Loop through each row in the CSV file to get the image file and bounding box info
for _, row in df.iterrows():
    # Get the image name and path
    image_name = row['image_name']  # Assuming 'image_name' column in CSV
    image_path = os.path.join(image_folder, image_name)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image file {image_name} not found. Skipping.")
        continue

    # Open the image to get its dimensions (width, height)
    image = Image.open(image_path)
    image_width, image_height = image.size  # Dynamic image size

    # Get the bounding box information
    xmin = row['x0']
    ymin = row['y0']
    xmax = row['x1']
    ymax = row['y1']
    
    # Convert the bounding box coordinates to YOLO format (normalized)
    x_center = (xmin + xmax) / 2 / image_width
    y_center = (ymin + ymax) / 2 / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    
    # Construct the label file path for each image
    label_file = os.path.join(label_folder, os.path.splitext(image_name)[0] + ".txt")
    
    # Write the label in YOLO format (class_id x_center y_center width height)
    with open(label_file, 'a') as f:
        f.write(f"0 {x_center} {y_center} {width} {height}\n")  # '0' is the class_id for face

print("Label files created successfully!")
