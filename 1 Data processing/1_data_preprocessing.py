import numpy as np
import pandas as pd
import random
import torch
import os
import sys
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split

cwd = os.getcwd() #cwd should be amld2025_fracAtlas
running_script = True # True if this script is run as a whole, False if it is run in parts (line by line)
print("cwd: ", cwd)

if running_script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Adjust depending on where 'functions.py' is
    sys.path.append(project_root)
    print("script dir:", script_dir)
    print("added to sys.path:", project_root)
else:
    # Add the root directory to the system path
    sys.path.append(os.path.abspath(os.path.join(cwd, '..')))
    
import functions

functions.test_func() #test function to check if the functions.py is working

# Define paths
fracAtlas = cwd + "\\Data\\fracAtlas\\" #path to fracAtlas folder
all_frac_folder = fracAtlas + "\\images\\all\\"
annotation_path = fracAtlas + "\\Annotations\\YOLO\\" #path to yolo annotations folder
fractured_path = fracAtlas + "\\images\\Fractured\\"
non_fractured_path = fracAtlas + "\\images\\Non_fractured\\"
preproc_folder = cwd + "\\Data\\data_rgb\\"
# Create output directories
output_dir = cwd + "\\Data\\split_data"
os.makedirs(output_dir, exist_ok=True)

# Define transformations
transform = functions.load_transformations(rgb=True,
                                           img_size= 416)

 
# Create train/val/test directories with images and labels subdirectories
for split in ['train', 'val', 'test']:
    for subdir in ['images', 'labels', 'ann_rcnn']:
        os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

#read dataset.xlsx
#contains all image names and some data about each image
df_meta = pd.read_csv(fracAtlas + 'dataset.csv')
df_meta = df_meta.rename(columns={'fractured': 'label'}) #rename the column to label
df_meta["img_path"] = df_meta["image_id"].apply(lambda x: os.path.join(all_frac_folder, x)) #add the image path to the dataframe
df_label = df_meta[['image_id','label', 'img_path']] #keep only the image name and label
df_meta.to_csv(fracAtlas + 'meta_data.csv', index=False) #save the dataframe to a csv file



# split the dataset into train, validation and test sets
# stratify based on the label column to ensure that each set has the same proportion of fractured and non-fractured images
df_dev, df_test = train_test_split(df_meta, test_size = 0.15, random_state=1, stratify=df_meta['label'])
df_train, df_val = train_test_split(df_dev, test_size = 0.20, random_state=1, stratify=df_dev['label'])
 
# Print dataset information
print(f"Number of fractured images: {len(df_label[df_label['label'] == 1])}")
print(f"Number of non-fractured images: {len(df_label[df_label['label'] == 0])}")
 
# Function to copy and resize images to the right place
def process_image_set(split_df, split_name,
                      target_size=(416, 416),
                      annotations_path = os.getcwd()+"\\Data\\fracAtlas\\Annotations\\"):
    #annotations paths
    annotations_path_yolo = os.path.join(annotations_path, 'YOLO')
    annotations_path_rcnn = os.path.join(annotations_path, 'PASCAL VOC')
    processed_count = 0
    skipped_count = 0

    image_set = split_df['img_path'].tolist()
    fractured_images = split_df[split_df['label'] == 1]['img_path'].tolist()

    for img_path in image_set:
        try:
            # Get image filename
            img_filename = os.path.basename(img_path)
            img_basename = os.path.splitext(img_filename)[0]
       
            # Resize and save image
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
                    txt_filename = f"{img_basename}.txt" #for yolo
                    xml_filename = f"{img_basename}.xml" #for rcnn
                    txt_path = os.path.join(annotations_path_yolo, txt_filename)
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
                    
                    # open xml file located in annotations_path_rcnn and copy it to the destination folder
                    xml_path = os.path.join(annotations_path_rcnn, xml_filename)
                    dst_xml_path = os.path.join(output_dir, split_name, 'ann_rcnn', xml_filename)
                    if os.path.exists(xml_path):
                        # Copy the XML file to the destination folder
                        with open(xml_path, 'r') as src_file:
                            xml_content = src_file.read()
                        with open(dst_xml_path, 'w') as dst_file:
                            dst_file.write(xml_content)
                    else:
                        print(f"Warning: XML file not found for image: {img_filename}")

               
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                skipped_count += 1
               
        except Exception as e:
            print(f"Unexpected error with {img_path}: {str(e)}")
            skipped_count += 1
   
    return processed_count, skipped_count

# Process each split with the improved function
print("Processing training images...")
train_processed, train_skipped = process_image_set(df_train, 'train')
 
print("Processing validation images...")
val_processed, val_skipped = process_image_set(df_val, 'val')
 
print("Processing test images...")
test_processed, test_skipped = process_image_set(df_test, 'test')

# Print results
print(f"\nTraining: {train_processed} processed, {train_skipped} skipped")
print(f"Validation: {val_processed} processed, {val_skipped} skipped")
print(f"Testing: {test_processed} processed, {test_skipped} skipped")


###########################################################################
##### Preprocessing of train, val and test
###########################################################################¨



data_split = functions.load_data(cwd = cwd,
                            return_torch = True,
                            batch_size = 32,
                            rgb = True)

train_dataset = data_split["train_dataset"]
val_dataset = data_split["val_dataset"]
test_dataset = data_split["test_dataset"]

