

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

def process_image_set(image_set, split_name):
    processed_count = 0
    skipped_count = 0
    
    for img_path in image_set:
        try:
            # Get image filename
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            
            # Define destination paths
            dst_img_path = os.path.join(output_dir, split_name, 'images', img_filename)
            
            # Try to open and resize the image
            try:
                with Image.open(img_path) as img:
                    # Resize image to target size
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(dst_img_path)
                    processed_count += 1
                    
                    # Check if this is a fractured image
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
                        # For non-fractured images: create an empty label file
                        with open(dst_txt_path, 'w') as f:
                            pass  # Just create an empty file
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                skipped_count += 1
                
        except Exception as e:
            print(f"Unexpected error with {img_path}: {str(e)}")
            skipped_count += 1
    
    return processed_count, skipped_count

# Process each split with the improved function
print("Processing training images...")
train_processed, train_skipped = process_image_set(train_images, 'train')

print("Processing validation images...")
val_processed, val_skipped = process_image_set(val_images, 'val')

print("Processing test images...")
test_processed, test_skipped = process_image_set(test_images, 'test')

# Print results
print(f"\nTraining: {train_processed} processed, {train_skipped} skipped")
print(f"Validation: {val_processed} processed, {val_skipped} skipped")
print(f"Testing: {test_processed} processed, {test_skipped} skipped")

sample_path = os.path.join(output_dir, 'train', 'images', os.listdir(os.path.join(output_dir, 'train', 'images'))[0])
with Image.open(sample_path) as img:
    print(f"Sample image dimensions: {img.size}")



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
model_non_pretrained = YOLO('yolo11n.yaml')  # load a non-pretrained model

# Train the model with 416x416 image size
results = model.train(
    data=os.path.join(output_dir, 'dataset.yaml'), 
    epochs=50, 
    imgsz= 416,
    batch = 16,# This sets the training image size to 416x416
    patience = 10
)


# Train the model with 416x416 image size
results = model_non_pretrained.train(
    data=os.path.join(output_dir, 'dataset.yaml'), 
    epochs=50, 
    imgsz= 416,
    batch = 16,# This sets the training image size to 416x416
    patience = 10
)



if __name__ == '__main__':
    # Load the trained model with properly formatted path
    model = YOLO(r"C:\Users\Soren\Documents\amld2025_fracAtlas\2. Train the models\runs\detect\train9_pretrained\weights\best.pt")

    # Validate the model on your test set
    metrics = model.val(data=os.path.join(output_dir, 'dataset.yaml'))
    print(f"Validation metrics: {metrics}")
