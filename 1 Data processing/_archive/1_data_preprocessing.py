'''
Author: Sebastian Faurby

Date: 31/03/2025

Description: First script for data pre-processing. This script reads the dataset, which is on a png format per image.
The purpose of the script is then to print all this data in a tensor format, which is the format that the neural network will use to train.

Notice that when running the script you need to ensure that your current working directory is the root folder of the repository (main folder).
'''

import numpy as np
import pandas as pd
import random
import torch
import os
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split

########################################################################
##### 0.1 Classes and functions 
########################################################################

#class to load the dataset
class FracDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform = None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'image_id' and 'label' columns
            image_folder (str): Path to folder containing all images
            transform (callable, optional): Transformations to apply on the image
        """
        self.dataframe = dataframe.reset_index(drop=True) #reset the index of the dataframe
        self.image_folder = image_folder
        self.ann_yolo_folder = ann_yolo_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get the image name and label from the dataframe
        image_name = self.dataframe.loc[idx, "image_id"]
        label = self.dataframe.loc[idx, "label"]

        # Full image path
        image_path = os.path.join(self.image_folder, image_name)

        # Load the image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # boxes
        boxes = load_yolo_annotations(image_name, self.ann_yolo_folder)

        # Apply transformation if any
        if self.transform:
            image = self.transform(image)

        return image, boxes, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((416,416)), #could 416,416 or 640 x 640 depending on the model
    transforms.ToTensor(), # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image with ImageNet stats
])


# Optional: Unnormalize image for proper display
def unnormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_tensor * std + mean

def plot_image(dataset, idx):
    image_tensor, box, label = dataset[idx]
    
    # Unnormalize for display
    image_tensor = unnormalize(image_tensor)
    
    # Convert from [C, H, W] to [H, W, C] for matplotlib
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    plt.imshow(image_np)
    plt.title(f"Label: {'Fractured' if label == 1 else 'Non-fractured'}")
    plt.axis('off')
    plt.show()

# function to load the annotations from the yolo format
def load_yolo_annotations(image_name, yolo_folder):
    '''
    Reads the YOLO annotation file corresponding to the image.
    Returns a list of bounding boxes [x_center, y_center, width, height].
    '''

    annotation_file = os.path.join(yolo_folder, image_name.replace('.jpg', '.txt'))
    boxes = []

    if os.path.exists(annotation_file) and os.path.getsize(annotation_file) > 0:
        with open(annotation_file, 'r') as f:
            for line in f:
                # Each lins is: class x_center y_center width height
                parts = line.strip().split()
                box  = list(map(float, parts[1:])) # no need for class label
                boxes.append(box)
    return boxes

def plot_img_with_boxes(dataset, idx, yolo_folder):
    '''
    Plots the image with the bounding boxes from the YOLO annotations.
    '''
    image_tensor, box, label = dataset[idx]

    # Unnormalize the image for display
    image_tensor = unnormalize(image_tensor)

    #Convert from [C, H, W] to [H, W, C] for matplotlib
    image_np = image_tensor.permute(1,2,0).numpy()

    # Load the corresponding annotation boxzes if any
    image_name = dataset.dataframe.loc[idx, 'image_id']
    boxes = load_yolo_annotations(image_name, yolo_folder)

    plt.imshow(image_np)
    plt.title(f"Label: {'Fractured' if label == 1 else 'Non-fractured'}")

    #if the are bounding boxes, draw them on the image
    if boxes:
        img_width, img_height = image_tensor.shape[1], image_tensor.shape[2]

        for box in boxes:
            x_center, y_center, width, height = box

            # Convert YOLO format to pixel coordinates
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            width = int(width * img_width)
            height = int(height * img_height)

            #compute the bouindding box corners
            xmin = int(x_center - width / 2)
            xmax = int(x_center + width / 2)
            ymin = int(y_center - height / 2)
            ymax = int(y_center + height / 2)

            # draw the bounding box
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), width, height, edgecolor = 'red', facecolor = 'none'))

    plt.axis('off')
    plt.show()

########################################################################
#####  0.2 Standard configurations
########################################################################

# define cwd
cwd = os.getcwd() #cwd should be amld2025_fracAtlas
fracAtlas = cwd + "\\Data\\fracAtlas\\" #path to fracAtlas folder
frac_folder = fracAtlas + "\\images\\Fractured\\"
non_frac_folder = fracAtlas + "\\images\\Non_fractured\\"
all_frac_folder = fracAtlas + "\\images\\all\\"
ann_yolo_folder = fracAtlas + "\\Annotations\\YOLO\\" #path to yolo annotations folder
preproc_folder = cwd + "\\Data\\data_rgb\\"

#read dataset.xlsx
#contains all image names and some data about each image
df_meta = pd.read_csv(fracAtlas + 'dataset.csv')
df_meta = df_meta.rename(columns={'fractured': 'label'}) #rename the column to label
df_label = df_meta[['image_id','label']].reset_index(drop=True) #keep only the image name and label

#split the dataset into fractured and non-fractured images
df_fractured = df_meta[df_meta['label'] == 1]
df_non_fractured = df_meta[df_meta['label'] == 0]

### Definition of global parameters
batch_size = 32 #batch size for the dataloader


########################################################################
#####  1.1 Load the data
########################################################################

##### Load images and labels as tensors

# split the dataset into train, validation and test sets
# stratify based on the label column to ensure that each set has the same proportion of fractured and non-fractured images
df_dev, df_test = train_test_split(df_meta, test_size = 0.2, random_state=1, stratify=df_meta['label'])
df_train, df_val = train_test_split(df_dev, test_size = 0.25, random_state=1, stratify=df_dev['label'])

# Create datasets
train_dataset = FracDataset(dataframe = df_train, image_folder=all_frac_folder, transform = transform)
val_dataset = FracDataset(dataframe = df_val, image_folder=all_frac_folder, transform = transform)
test_dataset = FracDataset(dataframe = df_test, image_folder=all_frac_folder, transform = transform)
#Ekstra dataset for fractured images only - not used in training
frac_dataset = FracDataset(dataframe=df_fractured, image_folder=frac_folder, transform=transform)

# lets check whether the dataset is working
img_tnsr, box, label = frac_dataset[200] #get the first image and label from the dataset
img_tnsr.shape

# create dataloaders for batch processing
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
# Extra dataloader for fractured images only - not used in training
frac_dataloader = DataLoader(frac_dataset, batch_size = batch_size, shuffle = False)

# # lets check whether the bunch of numbers (tensors) are correctly depicted as images
# plot_image(frac_dataset, 0) #plot the first image from the dataset

# ##### lets try annotations
# plot_img_with_boxes(frac_dataset, 5, ann_yolo_folder) #plot the first image with annotations

# for i in random.sample(range(len(test_dataset)), 5):
#     plot_img_with_boxes(frac_dataset, i, ann_yolo_folder) 


########################################################################
#####  1.2 Save the data
########################################################################

# After creating your datasets, save them as PyTorch files
torch.save(train_dataset, preproc_folder + '\\train_dataset.pt')
torch.save(val_dataset, preproc_folder + '\\val_dataset.pt')
torch.save(test_dataset, preproc_folder + '\\test_dataset.pt')



# Create a dictionary with all necessary dataloaders
data_package = {
    'train_dataloader': train_dataloader,
    'val_dataloader': val_dataloader,
    'test_dataloader': test_dataloader
}

# Save to disk
torch.save(data_package, preproc_folder + '\\dataloaders.pt')


# Load data
train_dataset = torch.load(preproc_folder + '\\train_dataset.pt')

# load dataloaders
data_package = torch.load(preproc_folder + '\\dataloaders.pt')
train_dataloader = data_package['train_dataloader']

img_tnsr, box, label = train_dataset[0] #get the first image and label from the dataset
print("Test print shape of image tensor: ", img_tnsr.shape)

print("Code terminated successfully")