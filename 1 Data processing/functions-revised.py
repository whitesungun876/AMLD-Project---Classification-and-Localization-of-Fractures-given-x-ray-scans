"Author: Jieyu Lian 04/22"

import numpy as np
import pandas as pd
import random
import torch
import os
import sys
from PIL import Image
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split

def load_yolo_annotations(image_name, yolo_folder):
    """
    Reads the YOLO annotation file corresponding to the image.
    Returns a list of bounding boxes [x_center, y_center, width, height].
    """
    annotation_file = os.path.join(yolo_folder, image_name.replace('.jpg', '.txt'))
    boxes = []

    if os.path.exists(annotation_file) and os.path.getsize(annotation_file) > 0:
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                box = list(map(float, parts[1:]))  # Skip class label, take only bbox data
                boxes.append(box)

    return boxes

# class to load the dataset
class FracDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, ann_yolo_folder=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'image_id' and 'label' columns
            image_folder (str): Path to folder containing all images
            transform (callable, optional): Transformations to apply on the image
            ann_yolo_folder (str): Path to YOLO annotations folder
        """
        self.dataframe = dataframe.reset_index(drop=True)  # reset the index of the dataframe
        self.image_folder = image_folder
        self.ann_yolo_folder = ann_yolo_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.loc[idx, "image_id"]
        label = self.dataframe.loc[idx, "label"]

        image_path = os.path.join(self.image_folder, image_name)

        image = Image.open(image_path).convert("RGB")

        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Image.DecompressionBombError):
            print(f"Warning: Skipping damaged image {image_name}")
            return None  

        boxes = []  

        if self.transform:
            image = self.transform(image)

        target = {}

        if not boxes:  
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)  
            target["labels"] = torch.tensor([0], dtype=torch.int64)  

        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            target["boxes"] = boxes_tensor
            target["labels"] = labels

        return image, target


# Function to load the dataset
def load_data(cwd=os.getcwd(), return_torch=True, batch_size=32, rgb=True):
    # Set path to FracAtlas data
    fracAtlas = os.path.join(cwd, "Data", "FracAtlas")
    print(f"Trying to load dataset from: {os.path.join(fracAtlas, 'dataset.csv')}")
    
    # Load the CSV file into a DataFrame
    df_meta = pd.read_csv(os.path.join(fracAtlas, "dataset.csv"))
    
    # Rename 'fractured' column to 'label' to match the expected column name
    df_meta = df_meta.rename(columns={'fractured': 'label'})

    # Set paths for images and annotations
    all_frac_folder = os.path.join(fracAtlas, "images", "all")
    annotation_path = os.path.join(fracAtlas, "Annotations", "YOLO")

    # Load transformations
    transform = load_transformations(rgb=True, img_size=416)

    # Split the dataset into training, validation, and test sets
    df_dev, df_test = train_test_split(df_meta, test_size=0.15, random_state=1, stratify=df_meta['label'])
    df_train, df_val = train_test_split(df_dev, test_size=0.20, random_state=1, stratify=df_dev['label'])

    # Create datasets using the FracDataset class
    train_dataset = FracDataset(dataframe=df_train, image_folder=all_frac_folder, transform=transform)
    val_dataset = FracDataset(dataframe=df_val, image_folder=all_frac_folder, transform=transform)
    test_dataset = FracDataset(dataframe=df_test, image_folder=all_frac_folder, transform=transform)

    # Create data loaders for batch processing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return the datasets and data loaders if return_torch is True
    if return_torch:
        data_split = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'test_dataloader': test_dataloader
        }
    else:
        # Return pandas split only if return_torch is False
        data_split = {
            'train_split': df_train,
            'val_split': df_val,
            'test_split': df_test,
        }

    return data_split



# Function to load transformations
def load_transformations(rgb=True, img_size=416):
    if rgb:
        # Define transformations for RGB images
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize image
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
    else:
        # Define transformations for grayscale images
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize image
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.456], std=[0.224])  # Normalize for grayscale images
        ])
    return transform

