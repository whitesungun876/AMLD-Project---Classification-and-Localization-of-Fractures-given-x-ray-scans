

"""
Prediction script for Faster R-CNN model
Loads trained weights and runs inference on test dataset
Outputs predictions to evaluation_results directory
Calculates and saves evaluation metrics (mAP50-95, mAP50, Precision, Recall, F1-Score)
Saves visualization of predicted vs ground truth bounding boxes
"""

import os
import csv
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import torchvision.transforms.functional as F

# ----------------------
# Configuration
# ----------------------
CWD = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CWD, "Data", "split_data", "test")
OUTPUT_DIR = os.path.join(CWD, "evaluation_results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 0
MIN_SIZE = 416
MAX_SIZE = 416

# Path to the weights
WEIGHTS_PATH = os.path.join(CWD, "weights", "fasterrcnn_best_loaded.pth")

# Confidence threshold for predictions
CONF_THRESHOLD = 0.4

# Evaluation parameters
IOU_THRESHOLD = 0.5  # for mAP50
IOU_RANGE = np.linspace(0.5, 0.95, 10)  # for mAP50-95
NUM_VISUALIZATIONS = 10  # Number of sample images to save with both GT and pred boxes

# ----------------------
# Dataset and Model Utils
# ----------------------
def collate_fn(batch):
    return tuple(zip(*batch))

class YOLODataset:
    def __init__(self, base_dir):
        base_dir = Path(base_dir)
        self.images = sorted((base_dir/"images").glob("*.[jp][pn]g"))
        self.labels_dir = base_dir/"labels"
        if not self.images:
            raise RuntimeError(f"No images found in {base_dir/'images'}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = read_image(str(img_path)).float() / 255.0
        _, H, W = img.shape
        
        # Load ground truth labels if available
        target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
        txt_path = self.labels_dir/f"{img_path.stem}.txt"
        
        if txt_path.exists():
            boxes, labels = [], []
            for line in txt_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:  # class x_center y_center width height
                    c = float(parts[0])
                    xc, yc = float(parts[1]) * W, float(parts[2]) * H
                    w, h = float(parts[3]) * W, float(parts[4]) * H
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1, y1 = xc - w/2, yc - h/2
                    x2, y2 = xc + w/2, yc + h/2
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c))
            
            if boxes:
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)
        
        return img, target, img_path.name, img_path

def make_model():
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,  # No need for pre-trained weights as we're loading our own
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=CONF_THRESHOLD,
        box_nms_thresh=0.5,
        box_detections_per_img=10,
    )

    anchor_gen = AnchorGenerator(
        sizes=((4,8,16,32,64),)*5,
        aspect_ratios=((0.25,0.5,1.0,2.0,4.0),)*5
    )
    model.rpn.anchor_generator = anchor_gen
    in_ch = model.backbone.out_channels
    num_anc = anchor_gen.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(in_ch, num_anc)

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    return model

# ----------------------
# Evaluation Metrics Functions
# ----------------------
def box_iou(box1, box2):
    """
    Compute IoU between boxes
    box format: [x1, y1, x2, y2]
    """
    # Handle empty boxes
    if box1.shape[0] == 0 or box2.shape[0] == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)
    
    # Get intersection rectangle coordinates
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
    
    # Calculate intersection area
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union area
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-10)  # Add epsilon to avoid division by zero
    return iou

def compute_ap(recalls, precisions):
    """
    Compute Average Precision using 11-point interpolation
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

def calculate_metrics(all_detections, all_targets, iou_threshold=0.5, score_threshold=None):
    """
    Calculate AP, Precision, Recall, and F1-Score
    
    Arguments:
        all_detections: list of detection dictionaries (boxes, scores, labels)
        all_targets: list of target dictionaries (boxes, labels)
        iou_threshold: IoU threshold for considering a detection as correct
        score_threshold: Optional confidence threshold to filter detections
    
    Returns:
        metrics: dictionary with AP, precision, recall, and F1-score
    """
    # Initialize lists to collect true positives, false positives, and scores
    true_positives = []
    false_positives = []
    scores = []
    total_gt = 0
    
    # Process each image
    for detection, target in zip(all_detections, all_targets):
        # Extract prediction data
        pred_boxes = detection['boxes']
        pred_scores = detection['scores']
        pred_labels = detection['labels']
        
        # Extract ground truth data
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # Count ground truth objects (ignoring background class if present)
        gt_count = sum(1 for label in gt_labels if label > 0)
        total_gt += gt_count
        
        # Skip if no predictions or ground truth
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Calculate IoU between predictions and ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        
        # Track which ground truth boxes are matched
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        
        # Sort detections by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        # Process each detection in order of confidence
        for idx in sorted_indices:
            pred_label = pred_labels[idx]
            pred_score = pred_scores[idx].item()
            
            # Skip background class or detections below score threshold
            if pred_label == 0 or (score_threshold is not None and pred_score < score_threshold):
                continue
                
            scores.append(pred_score)
            
            # Find ground truth boxes with matching class and sufficient IoU
            valid_matches = ((gt_labels == pred_label) & (ious[idx] >= iou_threshold) & ~matched_gt)
            
            if valid_matches.any():
                # Mark best match as used
                best_match = torch.argmax(ious[idx] * valid_matches.float())
                matched_gt[best_match] = True
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
    
    # Convert to numpy arrays for calculations
    true_positives = np.array(true_positives, dtype=np.float32)
    false_positives = np.array(false_positives, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # If no detections, return zeros
    if len(scores) == 0:
        return {
            'AP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_positives': total_gt,
            'total_detections': 0,
            'tp': 0,
            'fp': 0
        }
    
    # Sort by score
    indices = np.argsort(-scores)
    true_positives = true_positives[indices]
    false_positives = false_positives[indices]
    
    # Calculate cumulative values
    cum_true_positives = np.cumsum(true_positives)
    cum_false_positives = np.cumsum(false_positives)
    
    # Calculate precision and recall
    precisions = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
    recalls = cum_true_positives / (total_gt + 1e-10)
    
    # Add sentinel values for AP calculation
    precisions_for_ap = np.concatenate(([0.0], precisions, [0.0]))
    recalls_for_ap = np.concatenate(([0.0], recalls, [1.0]))
    
    # Ensure precision is non-increasing for each recall value
    for i in range(len(precisions_for_ap) - 2, -1, -1):
        precisions_for_ap[i] = max(precisions_for_ap[i], precisions_for_ap[i + 1])
    
    # Calculate AP
    ap = compute_ap(recalls_for_ap, precisions_for_ap)
    
    # Use the precision and recall values at the highest F1 score
    if len(precisions) > 0 and len(recalls) > 0:
        # Calculate F1 scores for each threshold
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        # Find index with highest F1 score
        best_idx = np.argmax(f1_scores)
        precision = precisions[best_idx]
        recall = recalls[best_idx]
        f1_score = f1_scores[best_idx]
    else:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
    
    # Count total true positives and false positives
    tp = int(np.sum(true_positives))
    fp = int(np.sum(false_positives))
    
    return {
        'AP': ap,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_positives': total_gt,
        'total_detections': len(scores),
        'tp': tp,
        'fp': fp
    }

def calculate_map50_95(all_detections, all_targets, iou_range):
    """
    Calculate mAP@50-95 (COCO metric)
    """
    aps = []
    for iou_threshold in iou_range:
        metrics = calculate_metrics(all_detections, all_targets, iou_threshold)
        aps.append(metrics['AP'])
    
    # Calculate mAP@50-95
    mAP = np.mean(aps)
    return mAP

# ----------------------
# Prediction Functions
# ----------------------
def draw_predictions(image, boxes, scores, labels, gt_boxes=None, gt_labels=None, threshold=CONF_THRESHOLD):
    """
    Draw prediction boxes and ground truth boxes on the image
    Predicted boxes: green
    Ground truth boxes: red
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    h, w = image_np.shape[:2]
    
    # Draw ground truth boxes first
    if gt_boxes is not None and gt_labels is not None:
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw ground truth box in red
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label_text = f"GT Class {label}"
            cv2.putText(image_np, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw prediction boxes
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw prediction box in green
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label and score
        label_text = f"Pred {label}: {score:.2f}"
        cv2.putText(image_np, label_text, (x1, y1 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_np

def draw_comparison_figure(image, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, threshold=CONF_THRESHOLD):
    """
    Create a more detailed matplotlib figure showing predictions vs ground truth
    """
    # Convert tensor image to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    
    # Draw ground truth boxes (red)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1
        
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"GT Class {label}", color='r', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw prediction boxes (green)
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score < threshold:
            continue
            
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1
        
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"Pred {label}: {score:.2f}", color='g', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title with IoU info if there are both predictions and ground truth
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        if iou_matrix.numel() > 0:
            max_iou = iou_matrix.max().item() if iou_matrix.numel() > 0 else 0
            ax.set_title(f"Predictions vs Ground Truth (Max IoU: {max_iou:.3f})")
    else:
        ax.set_title("Predictions vs Ground Truth")
        
    ax.axis('off')
    
    return fig

def save_yolo_format(img_path, boxes, scores, labels, img_width, img_height, threshold=CONF_THRESHOLD):
    """
    Save predictions in YOLO format
    Format: <class> <x_center> <y_center> <width> <height> <confidence>
    """
    filename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(OUTPUT_DIR, "labels", f"{filename}.txt")
    
    with open(output_path, 'w') as f:
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            
            x1, y1, x2, y2 = box.tolist()
            
            # Convert to YOLO format (center coordinates and dimensions)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # YOLO format
            f.write(f"{int(label)} {x_center} {y_center} {width} {height} {score}\n")

def save_json_predictions(predictions, output_path):
    """
    Save all predictions to a JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

def predict_and_evaluate():
    """
    Run prediction on test dataset, evaluate metrics, and save results
    """
    print(f"Loading model from {WEIGHTS_PATH}")
    model = make_model().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    print(f"Processing test images from {TEST_DIR}")
    dataset = YOLODataset(TEST_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "comparison"), exist_ok=True)
    
    # Store predictions and ground truth for evaluation
    all_predictions = {}
    all_detections = []
    all_targets = []
    
    # Store image data for visualization of positive samples
    positive_samples = []
    
    # Process all images
    with torch.no_grad():
        for images, targets, img_names, img_paths in tqdm(dataloader, desc="Predicting"):
            # Move images to device
            images = [img.to(DEVICE) for img in images]
            
            # Run model inference
            outputs = model(images)
            
            # Process outputs
            for i, (output, target, img_name, img_path) in enumerate(zip(outputs, targets, img_names, img_paths)):
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()
                
                # Get original image dimensions
                _, img_height, img_width = images[i].shape
                
                # Draw predictions on image (without ground truth for basic visualization)
                vis_image = draw_predictions(images[i].cpu(), boxes, scores, labels)
                vis_path = os.path.join(OUTPUT_DIR, "visualization", img_name)
                cv2.imwrite(vis_path, vis_image)
                
                # Save in YOLO format
                save_yolo_format(img_path, boxes, scores, labels, img_width, img_height)
                
                # Store data for metrics calculation
                detections = {
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                }
                all_detections.append(detections)
                all_targets.append(target)
                
                # Store for JSON output
                all_predictions[img_name] = {
                    "boxes": [[float(x) for x in box] for box in boxes.tolist()],
                    "scores": [float(s) for s in scores.tolist()],
                    "labels": [int(l) for l in labels.tolist()],
                    "gt_boxes": [[float(x) for x in box] for box in target["boxes"].tolist()],
                    "gt_labels": [int(l) for l in target["labels"].tolist()]
                }
                
                # Store images with ground truth for comparison visualization
                # Only store images that have ground truth objects (positive samples)
                if len(target["boxes"]) > 0:
                    positive_samples.append({
                        "image": images[i].cpu(),
                        "pred_boxes": boxes,
                        "pred_scores": scores,
                        "pred_labels": labels,
                        "gt_boxes": target["boxes"],
                        "gt_labels": target["labels"],
                        "name": img_name
                    })
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    
    # Calculate mAP@50
    metrics_50 = calculate_metrics(all_detections, all_targets, IOU_THRESHOLD)
    
    # Calculate mAP@50-95
    map50_95 = calculate_map50_95(all_detections, all_targets, IOU_RANGE)
    
    # Combine metrics
    metrics = {
        "mAP50": metrics_50["AP"],
        "mAP50-95": map50_95,
        "precision": metrics_50["precision"],
        "recall": metrics_50["recall"],
        "f1_score": metrics_50["f1_score"],
        "true_positives": metrics_50["tp"],
        "false_positives": metrics_50["fp"],
        "total_gt_positives": metrics_50["total_positives"],
        "total_detections": metrics_50["total_detections"]
    }