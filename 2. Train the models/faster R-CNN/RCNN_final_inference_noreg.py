"""
SEF

Inference script for Faster R-CNN model evaluation on test data.
This script loads a trained Faster R-CNN model and runs inference on test images,
calculating various metrics (mAP50, mAP50-95, precision, recall, F1) and 
visualizing predictions on fracture images.
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchmetrics.detection import MeanAveragePrecision

# Constants and paths

model_name = "fasterrcnn_final_noreg.pth"
regularization = "no_regularization"  # Change to "regularization" if needed
test_type = "test"

CWD = os.getcwd()
TEST_DIR = os.path.join(CWD, "Data", "split_data", "test")
MODEL_PATH = os.path.join(CWD, "3. Evaluation", "weights", model_name)
OUTPUT_DIR = os.path.join(CWD, "3. Evaluation", "inference_rcnn", regularization, test_type)  
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
MIN_SIZE = 416
MAX_SIZE = 416
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.5  # Threshold for visualization

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


# Define the dataset class for loading test data
class YOLOTestDataset(Dataset):
    """Dataset for YOLO-format object detection test data."""
    def __init__(self, base_dir):
        base_dir = Path(base_dir)
        self.image_dir = base_dir / "images"
        self.label_dir = base_dir / "labels"
        
        # Check paths
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")
            
        # Get image files
        self.image_files = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        )
        print(f"Found {len(self.image_files)} test images in {self.image_dir}")
        
        # Track fracture images for visualization
        self.fracture_images = []
        for img_path in self.image_files:
            label_path = self.label_dir / img_path.with_suffix('.txt').name
            if label_path.exists():
                for line in label_path.read_text().splitlines():
                    if line.strip() and line.split()[0] == '1':  # Class 1 (fracture)
                        self.fracture_images.append(img_path)
                        break
        print(f"Found {len(self.fracture_images)} images with fractures")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Keep original image for visualization
        orig_image = Image.open(str(img_path)).convert("RGB")
        
        # Normalized tensor for model
        image = read_image(str(img_path)).to(torch.float32) / 255.0
        _, h, w = image.shape
        
        # Get labels
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip(): continue
                c, xc, yc, wn, hn = map(float, line.split())
                xc, yc, wn, hn = xc*w, yc*h, wn*w, hn*h
                x0, y0 = xc - wn/2, yc - hn/2
                x1, y1 = xc + wn/2, yc + hn/2
                boxes.append([x0, y0, x1, y1])
                labels.append(int(c))
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        target = {
            "boxes": boxes, 
            "labels": labels, 
            "image_id": torch.tensor([idx])
        }
        
        return image, target, orig_image, str(img_path)

# Collate function for batching
def collate_fn(batch):
    images, targets, orig_images, paths = zip(*batch)
    return images, targets, orig_images, paths

# Create the model with the same architecture as during training
def create_model():
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,  # We'll load our trained weights
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
    )
    
    # Custom anchors (same as training)
    anchor_gen = AnchorGenerator(
        sizes=((8,16,32),)*5,
        aspect_ratios=((0.5,1.0,2.0),)*5
    )
    model.rpn.anchor_generator = anchor_gen
    in_channels = model.backbone.out_channels
    num_anchors = anchor_gen.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(in_channels, num_anchors)
    
    # 2-class predictor (background, fracture)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    
    return model

# Function to calculate F1 score
def calculate_f1(precision, recall):
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0

# Function to visualize predictions
def visualize_predictions(image, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, image_path, idx):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot ground truth boxes in green
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor='green', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1-5, f'GT: {label}', 
            bbox=dict(facecolor='green', alpha=0.5), 
            fontsize=10, color='white'
        )
    
    # Plot prediction boxes in red
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            plt.gca().add_patch(rect)
            plt.text(
                x1, y1+15, f'Pred: {label} ({score:.2f})', 
                bbox=dict(facecolor='red', alpha=0.5), 
                fontsize=10, color='white'
            )
    
    plt.title(f"Image: {os.path.basename(image_path)}")
    plt.axis('off')
    
    # Save the visualization
    output_path = os.path.join(OUTPUT_DIR, f"visualization_{idx}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    return output_path

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    
    # Create and load model
    model = create_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load test dataset
    test_dataset = YOLOTestDataset(TEST_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize metric calculator
    map_metric = MeanAveragePrecision(iou_type="bbox")
    
    # For calculating precision, recall manually
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    # For storing fracture images for visualization
    fracture_images = []
    
    # Run inference
    print(f"Running inference on {test_type} dataset...")
    with torch.no_grad():
        for images, targets, orig_images, img_paths in tqdm(test_loader):
            # Move images to device
            images = [img.to(DEVICE) for img in images]
            
            # Run inference
            predictions = model(images)
            
            # Move predictions to CPU for metric calculation
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            
            # Update metrics
            map_metric.update(predictions, targets)
            
            # Calculate TP, FP, FN for precision, recall, F1
            for pred, target, orig_image, img_path in zip(predictions, targets, orig_images, img_paths):
                # Check if this is a fracture image
                has_fracture = any(label.item() == 1 for label in target['labels'])
                
                if has_fracture:
                    fracture_images.append((orig_image, target, pred, img_path))
                
                # Calculate metrics
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_labels = pred['labels']
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                # Only count detections with score >= threshold
                keep = pred_scores >= CONFIDENCE_THRESHOLD
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                
                # Count ground truth objects
                total_gt += len(gt_boxes)
                
                # Count TP and FP based on IoU
                matched_gt = [False] * len(gt_boxes)
                
                for box_idx, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                    # Check if the prediction matches any ground truth box
                    max_iou = -1
                    max_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if matched_gt[gt_idx] or pred_label != gt_label:
                            continue
                        
                        # Calculate IoU
                        x1 = max(pred_box[0], gt_box[0])
                        y1 = max(pred_box[1], gt_box[1])
                        x2 = min(pred_box[2], gt_box[2])
                        y2 = min(pred_box[3], gt_box[3])
                        
                        if x2 > x1 and y2 > y1:
                            intersection = (x2 - x1) * (y2 - y1)
                            area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                            union = area1 + area2 - intersection
                            iou = intersection / union
                            
                            if iou > max_iou and iou >= 0.5:  # IoU threshold of 0.5
                                max_iou = iou
                                max_idx = gt_idx
                    
                    if max_idx >= 0:
                        # True positive
                        total_tp += 1
                        matched_gt[max_idx] = True
                    else:
                        # False positive
                        total_fp += 1
    
    # Calculate metrics
    map_results = map_metric.compute()
    
    # Calculate precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = calculate_f1(precision, recall)
    
    # Print results
    print("\nTest Results:")
    print(f"mAP50: {map_results['map_50'].item():.4f}")
    print(f"mAP50-95: {map_results['map'].item():.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1'],
        'Value': [
            map_results['map_50'].item(),
            map_results['map'].item(),
            precision,
            recall,
            f1
        ]
    })
    
    results_csv_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Metrics saved to {results_csv_path}")
    
    # Visualize 10 random fracture images (or all if fewer than 10)
    visualized_paths = []
    if fracture_images:
        # Shuffle and select up to 10 fracture images
        random.shuffle(fracture_images)
        selected_images = fracture_images[:min(10, len(fracture_images))]
        
        print(f"\nVisualizing {len(selected_images)} fracture images...")
        for idx, (image, target, pred, img_path) in enumerate(selected_images):
            # Get ground truth boxes and labels
            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()
            
            # Get prediction boxes, scores, and labels
            pred_boxes = pred['boxes'].numpy()
            pred_scores = pred['scores'].numpy()
            pred_labels = pred['labels'].numpy()
            
            # Visualize
            output_path = visualize_predictions(
                image, gt_boxes, gt_labels,
                pred_boxes, pred_scores, pred_labels,
                img_path, idx
            )
            visualized_paths.append(output_path)
        
        print(f"Visualizations saved to {OUTPUT_DIR}")
    else:
        print("No fracture images found in the test set.")
    
    # Save detailed per-class metrics
    class_metrics = {
        'map_per_class': map_results['map_per_class'].tolist(),
        'mar_100_per_class': map_results['mar_100_per_class'].tolist()
    }
    
    detailed_metrics_df = pd.DataFrame({
        'Class': ['Background', 'Fracture'],
        'AP': class_metrics['map_per_class'],
        'AR': class_metrics['mar_100_per_class']
    })
    
    detailed_metrics_path = os.path.join(OUTPUT_DIR, "class_metrics.csv")
    detailed_metrics_df.to_csv(detailed_metrics_path, index=False)
    print(f"Per-class metrics saved to {detailed_metrics_path}")

if __name__ == "__main__":
    CWD = os.getcwd()
    models = ["fasterrcnn_final_noreg.pth"]
    test_folders = ["test", "test_brightness_minus_50","test_brightness_plus_50",
    "test_contrast_minus_50", "test_contrast_plus_50"]
    reg_types = {
        models[0]: "no_regularization"#,
        # models[1]: "regularization"
    }

    for model_name in models:
        for test_type in tqdm(test_folders):
            print(f"Running evaluation for {model_name} on {test_type}")
            regularization = reg_types[model_name]
            TEST_DIR = os.path.join(CWD, "Data", "split_data", test_type)
            MODEL_PATH = os.path.join(CWD, "3. Evaluation", "weights", model_name)
            OUTPUT_DIR = os.path.join(CWD, "3. Evaluation", "inference_rcnn", regularization, test_type)  
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            main()