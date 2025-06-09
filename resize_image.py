import os
import cv2

# Path to your dataset images
input_folder = './face_detection_yolov8/dataset/images/train'  # Replace with your image folder path
output_folder = './face_detection_yolov8/dataset/images/resized_train'  # Folder to save resized images

# Desired image size
new_size = (640, 640)  # Resize to 640x640

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in the input folder
for image_name in os.listdir(input_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, new_size)

        # Save the resized image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, resized_image)

        print(f'Resized and saved {image_name} to {output_folder}')
