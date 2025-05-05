import numpy as np
import pandas as pd
import random
import torch
import os
import sys
from PIL import Image
from datetime import datetime
from glob import glob
from torch.utils.data import DataLoader #, Dataset
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.ops import box_iou
import matplotlib.patches as patches
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
cwd = os.getcwd() #cwd should be amld2025_fracAtlas
running_script = False # True if this script is run as a whole, False if it is run in parts (line by line)
print("cwd: ", cwd)

if running_script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Adjust depending on where 'functions.py' is
    sys.path.append(project_root)
    print("script dir:", script_dir)
    print("added to sys.path:", project_root)
else:
    # Add the root directory to the system path
    sys.path.append(os.path.abspath(os.path.join(cwd, '..')))
    
import functions

###################################################################################
#### Standard Configuration ####
###################################################################################

# Define paths
fracAtlas = cwd + "\\Data\\fracAtlas\\" #path to fracAtlas folder
all_frac_folder = fracAtlas + "\\images\\all\\"
annotation_path = fracAtlas + "\\Annotations\\YOLO\\" #path to yolo annotations folder
fractured_path = fracAtlas + "\\images\\Fractured\\"
non_fractured_path = fracAtlas + "\\images\\Non_fractured\\"
graphics_path = cwd + "\\graphics\\FRCNN\\"
rcnn_path = cwd + "\\3. Results\\FRCNN\\models\\"

#batch size
batch_size = 4

#save date and time in format YYYY-MM-DD_HH-MM
date = datetime.now().strftime("%Y-%m-%d_%H-%M")

###################################################################################
#### Load the data ####
###################################################################################

dfs = functions.load_data(cwd = cwd,
                            return_torch = False,
                            batch_size = 32,
                            rgb = True)
df_train = dfs["train_split"]
df_val = dfs["val_split"]
df_test = dfs["test_split"]

dfs = [df_train, df_val, df_test]
names = ["train", "val", "test"]


data_split = functions.load_data(cwd = cwd,
                            return_torch = True,
                            batch_size = 32,
                            rgb = True)

train_dataset = data_split["train_dataset"]
val_dataset = data_split["val_dataset"]
test_dataset = data_split["test_dataset"]


# we need to convert the data to a format that is compatible with faster-RCNN (thanks for nothing SEF)
train_dataset = functions.FracDatasetWrapper(train_dataset)
val_dataset = functions.FracDatasetWrapper(val_dataset)
test_dataset = functions.FracDatasetWrapper(test_dataset)


# Create dataloaders with collate_fn for variable sized targets
def collate_fn(batch):
    return tuple(zip(*batch))

# Create data loaders
train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

test_dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

def yolo_to_xyxy(yolo_box, img_width, img_height):
    x_center, y_center, width, height = yolo_box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return [x_min, y_min, x_max, y_max]

# Function to create the faster RCNN model

def get_faster_rcnn_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights = 'COCO_V1')


    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Training function

def train_model(model, train_dataloader, val_dataloader, num_epochs=10):
    # Move the model to the device (GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Move model to the right device
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma=0.1)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    #initialize training
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train() # Set the model to training mode
        epoch_loss = 0.0

        for images, targets in tqdm(train_dataloader):
            # Move to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Check if loss_dict is a dict or a scalar/tensor
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                # If it's already a scalar or tensor (which happens during training)
                losses = loss_dict
            losses = sum(loss for loss in loss_dict.values())

            # Compute total loss
            loss_value = losses.item()
            epoch_loss += loss_value

            # backward pass and optimize
            losses.backward()
            optimizer.step()

        # Update learning rate
        lr_scheduler.step()

        # Calculate the average loss for the epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Training loss: {avg_train_loss:.4f}")

        #validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_dataloader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets) ### The loss_dict wasnt re defined here, so it was using the one from the training loop. 
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

                """# Check if loss_dict is a dict or a scalar/tensor
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    # If it's already a scalar or tensor (which happens during training)
                    losses = loss_dict
                val_loss += losses.item()"""

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")

    # Return trained model and losses for plotting
    return model, train_losses, val_losses


# Plot the loss curves
def plot_loss(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(graphics_path, f"loss_curve_{date}.png"))
    plt.close()

# Evaluation function
def evaluate_model(model, dataloader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = list(image.to(device) for image in images)

            # Get predictions
            predictions = model(images)

            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    return all_predictions, all_targets


# Function to calculate metrics of performance evaluation

def calculate_metrics(all_predictions, all_targets, iou_threshold=0.5):
    # This is a simplified metric calculation
    # For comprehensive evaluation, consider using torchvision's COCO evaluation tools
    
    total_images = len(all_predictions)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(total_images):
        pred_boxes = all_predictions[i]['boxes'].cpu()
        pred_scores = all_predictions[i]['scores'].cpu()
        pred_labels = all_predictions[i]['labels'].cpu()
        
        target_boxes = all_targets[i]['boxes'].cpu()
        target_labels = all_targets[i]['labels'].cpu()
        
        # Consider only predictions with score > 0.5
        high_score_indices = torch.where(pred_scores > 0.5)[0]
        pred_boxes = pred_boxes[high_score_indices]
        pred_labels = pred_labels[high_score_indices]
        
        # For each ground truth box, find if there's a matching prediction
        matched_gt = torch.zeros(len(target_boxes), dtype=torch.bool)
        
        for pred_idx in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(target_boxes)):
                if matched_gt[gt_idx]:  # Skip already matched ground truths
                    continue
                    
                # Calculate IoU
                iou = box_iou(pred_boxes[pred_idx:pred_idx+1], target_boxes[gt_idx:gt_idx+1]).item()
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if we found a match
            if best_iou > iou_threshold and best_gt_idx != -1:
                # Check if labels match
                if pred_labels[pred_idx] == target_labels[best_gt_idx]:
                    true_positives += 1
                    matched_gt[best_gt_idx] = True
                else:
                    false_positives += 1
            else:
                false_positives += 1
        
        # Count false negatives (unmatched ground truths)
        false_negatives += (matched_gt == False).sum().item()
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Visualization function
def visualize_predictions(model, dataset, indices,
                        show = False, figsize=(12, 12)):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    fig, axs = plt.subplots(len(indices), 2, figsize=figsize)
    
    if len(indices) == 1:
        axs = axs.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            prediction = model([image.to(device)])
        
        # Convert image from normalized tensor to numpy for visualization
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Plot ground truth
        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Ground Truth")
        
        if 'boxes' in target and len(target['boxes']) > 0:
            for box in target['boxes'].cpu().numpy():
                rect = patches.Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1], 
                    linewidth=2, 
                    edgecolor='r', 
                    facecolor='none'
                )
                axs[i, 0].add_patch(rect)
        
        # Plot prediction
        axs[i, 1].imshow(image)
        axs[i, 1].set_title("Prediction")
        
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_scores = prediction[0]['scores'].cpu().numpy()
        
        for box, score in zip(pred_boxes, pred_scores):
            if score > 0.5:  # Only show high confidence predictions
                rect = patches.Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1], 
                    linewidth=2, 
                    edgecolor='g', 
                    facecolor='none'
                )
                axs[i, 1].add_patch(rect)
                axs[i, 1].text(
                    box[0], box[1], 
                    f'Score: {score:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.7)
                )
        
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(graphics_path, f"predictions_{date}.png"))
    plt.close(fig)


####################################################################################
##### MAIN EXCEUTION #####
####################################################################################

# Create and train the model
model = get_faster_rcnn_model(num_classes=2)  # 1 class (fractured) + background
model, train_losses, val_losses = train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=1)

# Plot the loss curves
plot_loss(train_losses, val_losses)

# Evaluate of test dataset
all_predictions, all_targets = evaluate_model(model, test_dataloader)  

# Calculate metrics
metrics = calculate_metrics(all_predictions, all_targets)
print(f"Test set metrics:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Now visualize some examples
visualize_predictions(model, test_dataset, indices=[0, 1, 2])

# Save the model
torch.save(model.state_dict(), os.path.join(f"faster_rcnn_model_{date}.pth"))
print(f"Model saved as faster_rcnn_model_{date}.pth")
