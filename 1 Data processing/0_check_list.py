'''
Author: Sebastian Faurby

Date: 31/03/2025

Description: First script for data pre-processing. This script reads the dataset, which is on a png format per image.
The purpose of the script is then to print all this data in a tensor format, which is the format that the neural network will use to train.

Notice that when running the script you need to ensure that your current working directory is the root folder of the repository (main folder).
'''

import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
from glob import glob

# define cwd
cwd = os.getcwd() #cwd should be amld2025_fracAtlas
fracAtlas = cwd + "\\fracAtlas\\" #path to fracAtlas folder
frac_folder = fracAtlas + "\\images\\Fractured\\"
non_frac_folder = fracAtlas + "\\images\\Non_fractured\\"

#read dataset.xlsx
#contains all image names and some data about each image
df = pd.read_csv(fracAtlas + 'dataset.csv')

#split the dataset into fractured and non-fractured images
df_fractured = df[df['fractured'] == 1]
df_non_fractured = df[df['fractured'] == 0]

#lets read the image names and labels and compare them to the ones in the dataset
img_paths = []
img_labels = []
img_names = []

#get path fractured images
for img_path in glob(frac_folder + '*.jpg'):
    img_name = img_path.split("\\")[-1] #get the image name
    img_names.append(img_name) #append the image name to the list
    img_paths.append(img_path)
    img_labels.append(1)

#get all non-fractured images
for img_path in glob(non_frac_folder + '*.jpg'):
    img_name = img_path.split("\\")[-1] #get the image name
    img_names.append(img_name) #append the image name to the list
    img_paths.append(img_path)
    img_labels.append(0)


# df_path = pd.DataFrame({'image_id':img_names,'image_path': img_paths})
# df_path = df_path.sort_values(by='image_id') #sort the dataframe by image name
#join the two dataframes on image_id
# df_meta = pd.merge(df_path, df_meta, on='image_id', how='inner')

df_check = pd.DataFrame({'image_id':img_names,'image_path': img_paths, 'label': img_labels})
df_check = df_check.sort_values(by='image_id') #sort the dataframe by image name
df_check1 = df_check[['image_id','label']].reset_index(drop=True) #keep only the image name and label
df_check2 = df[['image_id','fractured']].reset_index(drop=True).rename(columns={'fractured': 'label'}) #keep only the image name and path

#check if the two dataframes are equal. they have same column names already
df_check1.equals(df_check2) #this should return True if the two dataframes are equal

if df_check1.equals(df_check2):
    print("The two dataframes are equal")



