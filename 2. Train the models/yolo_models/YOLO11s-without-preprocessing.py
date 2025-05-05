

'''
Date: 13/04/2025

Description: Script for setting up the YOLOv11 model for localization and classfication of fractures in radiographs.

'''


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import yaml


cwd = os.getcwd() 
#cwd = os.path.dirname(cwd)#cwd should be amld2025_fracAtlas



# Create output directories
output_dir = cwd + "\\Data\\split_data"
os.makedirs(output_dir, exist_ok=True)


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

fracture_path = os.path.join(output_dir, 'dataset.yaml')

# Load a model
model = YOLO('yolo11s.pt')  # load a pretrained model
# model_non_pretrained = YOLO('yolo11n.yaml')  # load a non-pretrained model



# Define search space
search_space = {
    # Initial learning rate: controls the step size at each update.
    "lr0":          (1e-4,   5e-3),  

    # Final LR factor: scales lr0 down at the end of training. 0.1 → 0.5 means ending between 10% and 50% of your initial LR.
    # Floor effect on 0.1 
    "lrf":          (0.015,   0.2),  

    # L2 weight decay to prevent overfitting
    # No good results on 0.0005
    "weight_decay": (1e-5,   3e-4),

    # Classification‑loss weight: balances cls vs box losses, the larger the more classification is favored.
    # best results around 0.7
    "cls":          (0.5,    1.0),

    # Box‑loss weight: how much the model penalizes bounding‑box errors.
    # best results around 0.11
    "box":          (0.08,   0.16),

    # Distance‑Focal‑Loss weight: refines box‑regression quality, the larger the more it favors tight boxes.
    "dfl":          (0.5,    1.0),      # tighten around 0.76

    # Scale‑augmentation range: zoom jitter from 80% to 120% of original.
    "scale":        (0.05,   0.15),
    
    "degrees":      (0.0, 10.0),  # Rotation degrees,
    
    "shear":       (0.0, 2.0),  # Shear degrees,
    
    "fliplr":      (0.0, 0.5) # Horizontal flip probability
    
}


# Common tuning args
common_args = dict(
    data=fracture_path,
    optimizer="AdamW",
    space=search_space,
    val=True,
    device='cuda:0',
    half=True,
    workers=6
)



# Phase 1: coarse search at 320×320
phase1_dir = os.path.join("runs", "hyp_search_phase1_yolo11s")
model.tune(
    **common_args,
    epochs=8,
    imgsz=320,
    batch=32,
    iterations=200,
    save=True,
    plots=True,
    project="runs",
    name="hyp_search_phase1_yolo11s"
)




phase1_dir = os.path.join(cwd,"runs", "hyp_search_phase1_yolo11s2")

# Load the best hyperparameters from phase 1
with open(os.path.join(phase1_dir, "best_hyperparameters.yaml"), 'r') as f:
    coarse_hyp = yaml.safe_load(f)

# Define search space
search_space_2 = {
    # Initial learning rate: controls the step size at each update.
    "lr0":          (1e-6,   5e-3),  

    # Final LR factor: scales lr0 down at the end of training. 0.1 → 0.5 means ending between 10% and 50% of your initial LR.
    "lrf":          (0.0015,   0.3),  

    # L2 weight decay to prevent overfitting
    "weight_decay": (1e-6,   3e-3),

    # Classification‑loss weight: balances cls vs box losses, the larger the more classification is favored.
    "cls":          (0.4,    0.9),

    # Box‑loss weight: how much the model penalizes bounding‑box errors.
    # best results around 0.11
    "box":          (0.08,   0.25),

    # Distance‑Focal‑Loss weight: refines box‑regression quality, the larger the more it favors tight boxes.
    "dfl":          (0.3,    0.8),      # tighten around 0.76

    # Scale‑augmentation range: zoom jitter from 80% to 120% of original.
    "scale":        (0.02,   0.19),
    
    "fliplr":      (0.0, 0.5) # Horizontal flip probability
    
}

# Common tuning args
common_args_2nd = dict(
    data=fracture_path,
    optimizer="AdamW",
    space=search_space_2,
    val=True,
    device='cuda:0',
    half=True,
    workers=6
)


# Phase 2: refined search at 416×416, seeded with coarse_hyp

results_2 = model.tune(
    **common_args_2nd,
    **coarse_hyp,
    epochs=12,
    imgsz=416,
    batch=32,
    iterations=90, 
#    hyp=coarse_hyp,
    save=True,
    plots=True,
    project="runs",
    name="hyp_search_phase2_yolo11s"
)


phase2_dir = os.path.join(cwd,"runs", "hyp_search_phase2_yolo11s2")

# Load the best hyperparameters from phase 1
with open(os.path.join(phase2_dir, "best_hyperparameters.yaml"), 'r') as f:
    coarse_hyp_2 = yaml.safe_load(f)

#print("Best hyperparameters:", best_hyp)
"""

"""# Define search space
search_space_3 = {
    # Initial learning rate: controls the step size at each update.
    "lr0":          (1e-6,   5e-4),  

    # Final LR factor: scales lr0 down at the end of training. 0.1 → 0.5 means ending between 10% and 50% of your initial LR.
    "lrf":          (0.0015,   0.03),  

    # L2 weight decay to prevent overfitting
    "weight_decay": (1e-6,   3e-3),

    # Classification‑loss weight: balances cls vs box losses, the larger the more classification is favored.
    "cls":          (0.2,    0.6),

    # Box‑loss weight: how much the model penalizes bounding‑box errors.
    # best results around 0.11
    "box":          (0.08,   0.25),

    # Distance‑Focal‑Loss weight: refines box‑regression quality, the larger the more it favors tight boxes.
    "dfl":          (0.4,    0.9),      # tighten around 0.76

    # Scale‑augmentation range: zoom jitter from 80% to 120% of original.
    "scale":        (0.01,   0.12),
    
    "fliplr":      (0.15, 0.3), # Horizontal flip probability
    
    "dropout": (0.0, 0.5), # Dropout probability
    
}

# Common tuning args
common_args_3nd = dict(
    data=fracture_path,
    optimizer="AdamW",
    space=search_space_3,
    val=True,
    device='cuda:0',
    half=True,
    workers=6
)

results_3 = model.tune(
    **common_args_3nd,
    **coarse_hyp_2,
    epochs=10,
    imgsz=416,
    batch=32,
    iterations=40, 
    freeze = 9,
#    hyp=coarse_hyp,
    save=True,
    plots=True,
    project="runs",
    name="hyp_search_phase3_yolo11s"
)

phase3_dir = os.path.join(cwd,"runs", "hyp_search_phase3_yolo11s")

# Load the best hyperparameters from phase 3
with open(os.path.join(phase3_dir, "best_hyperparameters.yaml"), 'r') as f:
    coarse_hyp_3 = yaml.safe_load(f)
    
###########################################################
###### Train the model with the best hyperparameters ######
###########################################################

# Common training args
common_args_train = dict(
    data=fracture_path,
    optimizer="AdamW",
    val=True,
    device='cuda:0'
)


train_results = model.train(
    data=fracture_path,
    optimizer="AdamW",
    val=True,
    device='cuda:0',
    **coarse_hyp_3,
    epochs=200,
    imgsz=416,
    batch=32,
    freeze = 9, # Freezing backbone of the model since it is expected that the features are already learned, since the model is both pretrained and fine-tuned.
    # We are therefore looking for refine classification and localization of the fractures.
    save=True,
    plots=True,
    project="runs",
    name="train_results_yolo11s",   
)

"""
val_results = model.val(
    data=fracture_path,
    imgsz=416,
    iou=0.5,
    save=True,
    plots=True,
    project="runs",
    name="val_results_yolo11s_323"
)"""