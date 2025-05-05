

'''
Author: 

Date: 12/04/2025

Description: Script for setting up the YOLOv11 model for localization and classfication of fractures in radiographs.

'''

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import torch

# Define paths
annotation_path = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\Annotations\YOLO"
fractured_path = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\images\Fractured"
non_fractured_path = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\images\Non_fractured"

# Create output directories
output_dir = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset"
os.makedirs(output_dir, exist_ok=True)

# Create train/val/test directories with images and labels subdirectories
for split in ['train', 'val', 'test']:
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

# Get all image files
fractured_images = glob.glob(os.path.join(fractured_path, "*.jpg"))
non_fractured_images = glob.glob(os.path.join(non_fractured_path, "*.jpg"))
all_images = fractured_images + non_fractured_images

# Print dataset information
print(f"Number of fractured images: {len(fractured_images)}")
print(f"Number of non-fractured images: {len(non_fractured_images)}")

# Split data (70% train, 15% validation, 15% test)
train_images, temp_images = train_test_split(all_images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Target image size
target_size = (416, 416)

# Function to copy and resize images to the right place
def process_image_set(image_set, split_name):
    for img_path in image_set:
        # Get image filename
        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        
        # Resize and save image
        dst_img_path = os.path.join(output_dir, split_name, 'images', img_filename)
        
        # Open, resize, and save the image
        with Image.open(img_path) as img:

            # Resize image to 416x416
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_img.save(dst_img_path)
        
        # Check if this is a fractured image (by checking if it's in the fractured_images list)
        is_fractured = img_path in fractured_images
        
        # Handle annotation file
        txt_filename = f"{img_basename}.txt"
        txt_path = os.path.join(annotation_path, txt_filename)
        dst_txt_path = os.path.join(output_dir, split_name, 'labels', txt_filename)
        
        if is_fractured:
            # For fractured images: copy the annotation file if it exists
            if os.path.exists(txt_path):
                # Read the existing annotation file to adjust class index
                with open(txt_path, 'r') as src_file:
                    annotations = src_file.readlines()
                
                # Update the class index to 1 (fractured) and write to the destination
                with open(dst_txt_path, 'w') as dst_file:
                    for annotation in annotations:
                        parts = annotation.strip().split()
                        if len(parts) >= 5:  # ensure it's a valid annotation line
                            # Set class to 1 (fractured) and keep the bounding box coordinates
                            parts[0] = "1"  # Class 1 for fractured
                            dst_file.write(" ".join(parts) + "\n")
            else:
                print(f"Warning: Annotation file not found for fractured image: {img_filename}")
        else:
            # For non-fractured images: create an empty label file (no objects)
            # Note: We don't add any bounding boxes because there are no fractures to detect
            with open(dst_txt_path, 'w') as f:
                # Just create an empty file - no objects to detect
                pass

# Process each split
process_image_set(train_images, 'train')
process_image_set(val_images, 'val')
process_image_set(test_images, 'test')

# Create dataset.yaml file for YOLO
yaml_content = f"""
path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['non_fractured', 'fractured']
"""

with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"Dataset prepared for YOLO training at: {output_dir}")
print(f"All images resized to {target_size[0]}x{target_size[1]}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")


# Load a model
model = YOLO('yolo11n.pt')  # load a pretrained model


# Train the model with 416x416 image size
results = model.train(
    data=os.path.join(output_dir, 'dataset.yaml'), 
    epochs=3, 
    imgsz= 416,
    batch = 16,# This sets the training image size to 416x416
    patience = 10

)

if __name__ == '__main__':
    # Load the trained model with properly formatted path
    model = YOLO(r"C:\Users\Soren\Documents\amld2025_fracAtlas\2. Train the models\runs\detect\train\weights\best.pt")

    # Validate the model on your test set
    metrics = model.val(data=os.path.join(output_dir, 'dataset.yaml'))
    print(f"Validation metrics: {metrics}")
    
    # Predict on a single image
    results = model.predict(r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\test\images\IMG0000010.jpg")
    


