import numpy as np
import pandas as pd
import random
import torch
import os
import sys
from PIL import Image, ImageEnhance
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split

# Set base path relative to the script location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

# Import functions_revised from the root directory
import functions_revised as functions

# Test function to check if the module is working (comment out if not present)
# functions.test_func()

# Define paths
fracAtlas = os.path.join(base_dir, "Data", "FracAtlas")
all_frac_folder = os.path.join(fracAtlas, "images", "all")
annotation_path = os.path.join(fracAtlas, "Annotations", "YOLO")
fractured_path = os.path.join(fracAtlas, "images", "Fractured")
non_fractured_path = os.path.join(fracAtlas, "images", "Non_fractured")
preproc_folder = os.path.join(base_dir, "Data", "data_rgb")
output_dir = os.path.join(base_dir, "Data", "split_data")
os.makedirs(output_dir, exist_ok=True)

dataset_path = os.path.join(fracAtlas, 'dataset.csv')
df_meta = pd.read_csv(dataset_path)
df_meta = df_meta.rename(columns={'fractured': 'label'})
df_meta["img_path"] = df_meta["image_id"].apply(lambda x: os.path.join(all_frac_folder, x))
df_label = df_meta[['image_id','label', 'img_path']]
df_meta.to_csv(os.path.join(fracAtlas, 'meta_data.csv'), index=False)

# Define transformations
transform = functions.load_transformations(rgb=True, img_size=416)

# Create train/val/test directories with images and labels subdirectories
for split in ['train', 'val', 'test']:
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

# Split data
df_dev, df_test = train_test_split(df_meta, test_size=0.15, random_state=1, stratify=df_meta['label'])
df_train, df_val = train_test_split(df_dev, test_size=0.20, random_state=1, stratify=df_dev['label'])

def augment_and_save_image(image, output_base_dir, img_basename, is_fractured, annotation_lines):
    augmentations = {
        'rotate': image.rotate(15),
        'flip': image.transpose(Image.FLIP_LEFT_RIGHT),
        'bright': ImageEnhance.Brightness(image).enhance(1.5)
    }
    for aug_name, aug_image in augmentations.items():
        aug_filename = f"{img_basename}_{aug_name}.jpg"
        aug_path = os.path.join(output_base_dir, 'images', aug_filename)
        aug_image.save(aug_path)

        aug_label_path = os.path.join(output_base_dir, 'labels', f"{img_basename}_{aug_name}.txt")
        with open(aug_label_path, 'w') as f:
            if is_fractured and annotation_lines:
                for annotation in annotation_lines:
                    parts = annotation.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "1"
                        f.write(" ".join(parts) + "\n")
            else:
                pass

def process_image_set(split_df, split_name, target_size=(416, 416), annotations_path=annotation_path):
    processed_count = 0
    skipped_count = 0

    image_set = split_df['img_path'].tolist()
    fractured_images = split_df[split_df['label'] == 1]['img_path'].tolist()

    for img_path in image_set:
        try:
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
            dst_img_path = os.path.join(output_dir, split_name, 'images', img_filename)

            with Image.open(img_path) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(dst_img_path)
                processed_count += 1

                is_fractured = img_path in fractured_images
                txt_filename = f"{img_basename}.txt"
                txt_path = os.path.join(annotations_path, txt_filename)
                dst_txt_path = os.path.join(output_dir, split_name, 'labels', txt_filename)

                annotation_lines = None
                if is_fractured and os.path.exists(txt_path):
                    with open(txt_path, 'r') as src_file:
                        annotation_lines = src_file.readlines()
                    with open(dst_txt_path, 'w') as dst_file:
                        for annotation in annotation_lines:
                            parts = annotation.strip().split()
                            if len(parts) >= 5:
                                parts[0] = "1"
                                dst_file.write(" ".join(parts) + "\n")
                else:
                    with open(dst_txt_path, 'w') as f:
                        pass

                if split_name == 'train':
                    augment_and_save_image(resized_img, os.path.join(output_dir, split_name), img_basename, is_fractured, annotation_lines)

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            skipped_count += 1

    return processed_count, skipped_count

print("Processing training images...")
train_processed, train_skipped = process_image_set(df_train, 'train')
print("Processing validation images...")
val_processed, val_skipped = process_image_set(df_val, 'val')
print("Processing test images...")
test_processed, test_skipped = process_image_set(df_test, 'test')

print(f"\nTraining: {train_processed} processed, {train_skipped} skipped")
print(f"Validation: {val_processed} processed, {val_skipped} skipped")
print(f"Testing: {test_processed} processed, {test_skipped} skipped")

###########################################################################
##### Preprocessing of train, val and test
###########################################################################

data_split = functions.load_data(cwd=base_dir, return_torch=True, batch_size=32, rgb=True)
train_dataset = data_split["train_dataset"]
val_dataset = data_split["val_dataset"]
test_dataset = data_split["test_dataset"]
