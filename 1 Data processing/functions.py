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

#class to load the dataset
class FracDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform = None,
                 ann_yolo_folder = os.getcwd()+"\\Data\\fracAtlas\\Annotations\\YOLO\\"):
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




# Wrapper function around the FracDataset class to get a propper format for faster-RCNN

class FracDatasetWrapper(Dataset):
    def __init__(self, dataset, label_only = False,  yolo_to_rcnn = True):
        self.dataset = dataset
        self.label_only = label_only
        self.yolo_to_rcnn = yolo_to_rcnn

    def __getitem__(self, idx):
        # get the image and annotations
        image = self.dataset[idx].__getitem__(0) #Image tensor
        boxes = self.dataset[idx].__getitem__(1) #list of box coordinates
        label = self.dataset[idx].__getitem__(2) #binary label (0 or 1)

        # convert the format from yolo to pixel coordinates (x1,y1,x2,y2)
        # Suported by Faster R-CNN
        if self.yolo_to_rcnn:
            for i in range(len(boxes)):
                x_center, y_center, width, height = boxes[i]

                # Convert YOLO format to pixel coordinates
                x_center = int(x_center * image.shape[1])
                y_center = int(y_center * image.shape[2])
                width = int(width * image.shape[1])
                height = int(height * image.shape[2])

                # compute the bounding box corners
                xmin = int(x_center - width / 2)
                xmax = int(x_center + width / 2)
                ymin = int(y_center - height / 2)
                ymax = int(y_center + height / 2)

                boxes[i] = [xmin, ymin, xmax, ymax]

        # If it's kist a label dataset (no fracture, no annotations), return simplified format
        if self.label_only:
            return image, label
        
        # We also reformat the target to be compatible with the Faster R-CNN model
        target = {}

        if not boxes: #No fracture, no annotations, no boxes
            #We still give a format, but with empty tensor
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32) #empty tensor
            target["labels"] = torch.tensor([0], dtype=torch.int64)
        else:
            # Convert boxes to tensor format [x1,y2,x2,y2]
            boxes_tensor = torch.as_tensor(boxes, dtype = torch.float32)

            #All boxes have class 1 (fractured)
            labels = torch.ones((len(boxes),), dtype = torch.int64)

            target["boxes"] = boxes_tensor
            target["labels"] = labels

        # Note: We don't need to normalize boxes since we assume they're already
        # in the correct format relative to the resized 416x416 images

        return image, target
    
    def __len__(self):
        return len(self.dataset)

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

def load_transformations(rgb = True, img_size = 416):
    if rgb:
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size)), #could 416,416 or 640 x 640 depending on the model
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the image with ImageNet stats
        ])
    else:
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size)), #could 416,416 or 640 x 640 depending on the model
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize(mean=[0.456], std=[0.224]) # Normalize the image with grayscale mean and std
        ])
    return transform

def load_data(cwd = os.getcwd(),
            return_torch = True,
            batch_size = 32,
            rgb = True):
    # Define paths
    fracAtlas = cwd + "\\Data\\fracAtlas\\" #path to fracAtlas folder
    all_frac_folder = fracAtlas + "\\images\\all\\"
    annotation_path = fracAtlas + "\\Annotations\\YOLO\\" #path to yolo annotations folder

    # Define transformations
    transform = load_transformations(rgb=True,
                                            img_size= 416)

    #read dataset.xlsx
    #contains all image names and some data about each image
    df_meta = pd.read_csv(fracAtlas + 'dataset.csv')
    df_meta = df_meta.rename(columns={'fractured': 'label'}) #rename the column to label
    df_meta["img_path"] = df_meta["image_id"].apply(lambda x: os.path.join(all_frac_folder, x)) #add the image path to the dataframe

    # Define transformations
    transform = load_transformations(rgb=True, img_size= 416)

    # split the dataset into train, validation and test sets
    # stratify based on the label column to ensure that each set has the same proportion of fractured and non-fractured images
    df_dev, df_test = train_test_split(df_meta, test_size = 0.15, random_state=1, stratify=df_meta['label'])
    df_train, df_val = train_test_split(df_dev, test_size = 0.20, random_state=1, stratify=df_dev['label'])

    # Create datasets
    train_dataset = FracDataset(dataframe = df_train, image_folder=all_frac_folder, transform = transform)
    val_dataset = FracDataset(dataframe = df_val, image_folder=all_frac_folder, transform = transform)
    test_dataset = FracDataset(dataframe = df_test, image_folder=all_frac_folder, transform = transform)

    # create dataloaders for batch processing
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

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
        #return pandas split only
        data_split = {
            'train_split': df_train,
            'val_split': df_val,
            'test_split': df_test,
        }

    return data_split

def load_meta_data(cwd = os.getcwd()):
    meta_folder = cwd+"\\Data\\fracAtlas\\"
    #read meta_data.csv
    df = pd.read_csv(meta_folder + 'meta_data.csv')

    return df

def test_func():

    print("Test scripts is working succesfully")

    # return nothing
    return None