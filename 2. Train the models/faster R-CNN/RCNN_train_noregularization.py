# Script to load YOLO-format data and train Faster R-CNN with speed optimizations
"""
SEF

Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train & validate a Faster R-CNN ResNet50 FPN model.

Changes to RCNN_soe_tuning.py: this code will solely train the final model after hyperparameter tuning.
The model will be trained on the best hyperparameters found in the tuning script.
"""

import os
import random
from pathlib import Path
import joblib  # Add joblib for saving Optuna studies
import sys
import pandas as pd
import torch
# Enable cudnn autotune for optimal conv algorithms
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision as mAP
from torchvision.io import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import optuna
from optuna import TrialPruned
from optuna.pruners import MedianPruner  # Add MedianPruner for better trial efficiency
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # Import cosine scheduler
from torchvision import transforms as T
import torchvision.transforms.functional as F

# Check torch.compile availability with robust error handling
has_compile = False
try:
    if sys.version_info < (3, 12) and hasattr(torch, 'compile'):
        has_compile = True
    else:
        print("PyTorch compilation (torch.compile) not available:")
        if sys.version_info >= (3, 12):
            print("- Python 3.12+ detected: Dynamo not supported on Python 3.12+")
        elif not hasattr(torch, 'compile'):
            print("- torch.compile not found: Using PyTorch version without compilation support")
except Exception as e:
    print(f"Error checking torch.compile availability: {e}")
    has_compile = False

# Constants - Fix path handling
# Get the absolute path to the current working directory
CWD = os.getcwd()
TRAIN_DIR = os.path.join(CWD, "Data", "split_data", "train")
VAL_DIR = os.path.join(CWD, "Data", "split_data", "val")

print(f"Working directory: {CWD}")
print(f"Train directory: {TRAIN_DIR}")
print(f"Validation directory: {VAL_DIR}")

# Force single-process data loading to avoid multiprocessing issues
NUM_WORKERS = 0
print(f"Using {NUM_WORKERS} workers for data loading (multiprocessing disabled)")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH = 16
FINAL_EPOCHS = 200     # Number of epochs for final training
ACCUM_STEPS = 4
MIN_SIZE = 416
MAX_SIZE = 416

# Collate fn for variable-size targets
def collate_fn(batch):
    return tuple(zip(*batch))

# 1) Define train-time augmentations
def train_transform(image, target):
    # Random horizontal flip
    if random.random() < 0.5:
        _, H, W = image.shape
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0,2]] = W - boxes[:, [2,0]]
        target["boxes"] = boxes

    # Color jitter (brightness/contrast only for grayscale)
    pil = T.ToPILImage()(image)
    pil = T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.0,
        hue=0.0
    )(pil)
    image = T.ToTensor()(pil)
    return image, target

def calculate_map_precision_recall(all_detections, all_ground_truth, class_detections, class_ground_truth, iou_threshold=0.5):
    """
    Calculate mAP, precision, and recall metrics for object detection evaluation.
    
    Args:
        all_detections: List of dictionaries with detection results
        all_ground_truth: List of dictionaries with ground truth data
        class_detections: Dictionary with class-specific detections
        class_ground_truth: Dictionary with class-specific ground truth
        iou_threshold: IoU threshold for considering a detection as correct (default: 0.5)
        
    Returns:
        mAP: Mean Average Precision across all classes
        precision: Overall precision
        recall: Overall recall
    """
    import numpy as np
    
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        # Get coordinates of intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate area of intersection and union
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection = width * height
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Calculate IoU
        if union > 0:
            return intersection / union
        return 0
    
    def compute_ap(recall, precision):
        """Compute Average Precision using the 11-point interpolation"""
        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        return ap
    
    # Initialize metrics
    class_aps = {}
    total_tp = 0
    total_fp = 0
    total_gt = 0

    # Calculate AP for each class
    for class_id in class_detections.keys():
        # Skip if no ground truth for this class
        if class_id not in class_ground_truth:
            continue
            
        # Get all detections and ground truth for this class
        dets = class_detections[class_id]
        gts = class_ground_truth[class_id]
        
        # Count total ground truth objects
        num_gts = sum(len(gt['boxes']) for gt in gts)
        total_gt += num_gts
        
        if num_gts == 0:
            continue
            
        # Create a flat list of all detections
        all_dets = []
        for img_idx, det in enumerate(dets):
            for box_idx, (box, score) in enumerate(zip(det['boxes'], det['scores'])):
                all_dets.append({
                    'image_id': det['image_id'],
                    'box': box,
                    'score': score
                })
        
        # Sort detections by decreasing confidence
        all_dets = sorted(all_dets, key=lambda x: x['score'], reverse=True)
        
        # Initialize true/false positives arrays
        tp = np.zeros(len(all_dets))
        fp = np.zeros(len(all_dets))
        
        # Mark used ground truth boxes
        used_gt = {gt['image_id']: np.zeros(len(gt['boxes']), dtype=bool) for gt in gts}
        
        # Process each detection
        for det_idx, det in enumerate(all_dets):
            # Get ground truth for this image
            img_gts = next((gt for gt in gts if gt['image_id'] == det['image_id']), None)
            
            if img_gts is None or len(img_gts['boxes']) == 0:
                # No ground truth for this image, mark as false positive
                fp[det_idx] = 1
                continue
                
            # Calculate IoU with all ground truth boxes
            ious = [calculate_iou(det['box'], gt_box) for gt_box in img_gts['boxes']]
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            # Check if detection matches a ground truth
            if max_iou >= iou_threshold and not used_gt[det['image_id']][max_iou_idx]:
                # Mark as true positive and mark ground truth as used
                tp[det_idx] = 1
                used_gt[det['image_id']][max_iou_idx] = True
            else:
                # Mark as false positive
                fp[det_idx] = 1
        
        # Accumulate true/false positives
        total_tp += np.sum(tp)
        total_fp += np.sum(fp)
        
        # Compute precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        rec = cumsum_tp / num_gts
        prec = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # Calculate AP for this class
        ap = compute_ap(rec, prec)
        class_aps[class_id] = ap
    
    # Calculate mAP (mean of all class APs)
    if len(class_aps) > 0:
        mAP = np.mean(list(class_aps.values()))
    else:
        mAP = 0.0
    
    # Calculate overall precision and recall
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0
        
    if total_gt > 0:
        recall = total_tp / total_gt
    else:
        recall = 0.0
    
    return mAP, precision, recall

class YOLODataset(Dataset):
    """Dataset for YOLO-format object detection data."""
    def __init__(self, base_dir, transforms=None):
        base_dir = Path(base_dir)
        self.image_dir = base_dir / "images"
        self.label_dir = base_dir / "labels"
        
        # Add path existence check with better error messages
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")
            
        self.transforms = transforms
        
        # Keep using Path objects since we're not using multiprocessing
        self.image_files = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        )
        print(f"[YOLODataset] Found {len(self.image_files)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_image(str(img_path)).to(torch.float32) / 255.0
        _, h, w = image.shape
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
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

def make_model(w_box, w_rpn):
    """Faster R-CNN V2 with COCO-V2 weights, custom anchors, and 2-class head."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE,  # Use the constant
        max_size=MAX_SIZE,  # Use the constant
        
    )
    # Custom anchors
    anchor_gen = AnchorGenerator(
        sizes=((8,16,32),)*5,
        aspect_ratios=((0.5,1.0,2.0),)*5
    )
    model.rpn.anchor_generator = anchor_gen
    in_channels = model.backbone.out_channels
    num_anchors = anchor_gen.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(in_channels, num_anchors)
    # 2-class predictor
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    return model

class ModelWithCustomForward(torch.nn.Module):
    def __init__(self, base_model, w_cls, w_box, w_obj, w_rpn):
        super().__init__()
        self.base_model = base_model
        self.w_cls  = w_cls
        self.w_box  = w_box
        self.w_obj  = w_obj
        self.w_rpn  = w_rpn

    def forward(self, images, targets=None):
        if targets is not None:
            # ensure model is in train for loss
            training = self.base_model.training
            if not training: self.base_model.train()
            loss_dict = self.base_model(images, targets)
            if not training: self.base_model.eval()

            # apply weights
            total_loss = (
                  self.w_cls  * loss_dict['loss_classifier']
                + self.w_box  * loss_dict['loss_box_reg']
                + self.w_obj  * loss_dict['loss_objectness']
                + self.w_rpn  * loss_dict['loss_rpn_box_reg']
            )
            loss_dict['loss_total'] = total_loss
            return loss_dict

        return self.base_model(images)

def train_one_epoch(model, optimizer, loader, device, epoch, writer, scaler):
    model.train()
    optimizer.zero_grad()
    for step, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Train E{epoch}")):
        images = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k,v in t.items()} for t in tgts]
        with autocast():
            loss_dict = model(images, targets)
            loss = loss_dict['loss_total'] / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step+1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        gs = epoch * len(loader) + step
        writer.add_scalar('train/total_loss', loss_dict['loss_total'].item(), gs)
        for k, v in loss_dict.items():
            writer.add_scalar(f"train/{k}", v.item(), gs)

def validate(model, loader, device, writer, epoch):
    model.eval()
    sums = {k:0.0 for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg','loss_total']}
    cnt = 0
    with torch.no_grad():
        for step, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Val E{epoch}")):
            images = [img.to(device) for img in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            loss_dict = model(images, targets)
            for k, v in loss_dict.items():
                sums[k] += v.item()
                writer.add_scalar(f"val/{k}", v.item(), epoch*len(loader)+step)
            cnt += 1
    avg = {k: sums[k]/cnt for k in sums}
    print(f"[E{epoch}] Val cls:{avg['loss_classifier']:.4f} box:{avg['loss_box_reg']:.4f} obj:{avg['loss_objectness']:.4f} rpn:{avg['loss_rpn_box_reg']:.4f} total:{avg['loss_total']:.4f}")
    return avg

def val_results(
        model, loader,
        device, writer,
        epoch, df_val,
        type_split='val',
        score_threshold=0.05, iou_threshold=0.5
):

    from collections import defaultdict

    #evaluation mode
    model.eval()

    #loss accumulation
    sums = {k:0.0 for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg','loss_total']}
    cnt = 0

    # Detection metrics accumulators
    all_detections = []
    all_ground_truth = []
    
    # Class-specific metrics
    class_detections = defaultdict(list)
    class_ground_truth = defaultdict(list)

    with torch.no_grad():
        for step, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Eval: {type_split} E{epoch}")):
            # Move images and targets to the device
            images = [img.to(device) for img in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in tgts]

            # Forward pass through the model
            loss_dict = model(images, targets)

            # accumulate losses
            for k, v in loss_dict.items():
                sums[k] += v.item()
                writer.add_scalar(f"val/{k}", v.item(), epoch*len(loader)+step)
            cnt += 1

            # Get model predictions (forward pass without targets returns predictions)
            predictions = model.base_model(images)

            # Process each image in the batch
            for idx, (prediction, target) in enumerate(zip(predictions, targets)):
                # Filter predictions by confidence score
                keep = prediction['scores'] >= score_threshold
                pred_boxes = prediction['boxes'][keep]
                pred_scores = prediction['scores'][keep]
                pred_labels = prediction['labels'][keep]
                
                # Get ground truth
                gt_boxes = target['boxes']
                gt_labels = target['labels']
                
                # Store detections and ground truth for mAP calculation
                img_id = target['image_id'].item()
                
                # Store detections and ground truth for this image
                all_detections.append({
                    'image_id': img_id,
                    'boxes': pred_boxes.cpu().numpy(),
                    'scores': pred_scores.cpu().numpy(),
                    'labels': pred_labels.cpu().numpy()
                })
                
                all_ground_truth.append({
                    'image_id': img_id,
                    'boxes': gt_boxes.cpu().numpy(),
                    'labels': gt_labels.cpu().numpy()
                })
                
                # Store class-specific detections and ground truth
                for label in torch.unique(torch.cat([pred_labels, gt_labels])):
                    label_idx = label.item()
                    
                    # Class-specific predictions
                    class_pred_mask = pred_labels == label
                    class_detections[label_idx].append({
                        'image_id': img_id,
                        'boxes': pred_boxes[class_pred_mask].cpu().numpy(),
                        'scores': pred_scores[class_pred_mask].cpu().numpy()
                    })
                    
                    # Class-specific ground truth
                    class_gt_mask = gt_labels == label
                    class_ground_truth[label_idx].append({
                        'image_id': img_id,
                        'boxes': gt_boxes[class_gt_mask].cpu().numpy()
                    })
    
    # calculate average losses
    avg = {k: sums[k]/cnt for k in sums}
    print(f"[E{epoch}] Val cls:{avg['loss_classifier']:.4f} box:{avg['loss_box_reg']:.4f} obj:{avg['loss_objectness']:.4f} rpn:{avg['loss_rpn_box_reg']:.4f} total:{avg['loss_total']:.4f}")
    # Calculate mAP, precision, and recall
    mAP, precision, recall = calculate_map_precision_recall(
        all_detections, all_ground_truth, class_detections, 
        class_ground_truth, iou_threshold
    )
    print(f"[E{epoch}] mAP: {mAP:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    # Create DataFrame with all metrics
    df_val_new = pd.DataFrame({
        "epoch": epoch,
        "loss_classifier": avg['loss_classifier'],
        "loss_box_reg": avg['loss_box_reg'],
        "loss_objectness": avg['loss_objectness'],
        "loss_rpn_box_reg": avg['loss_rpn_box_reg'],
        "loss_total": avg['loss_total'],
        "mAP": mAP,
        "precision": precision,
        "recall": recall
    }, index=[0])

    # concat df_val_new with df_val
    df_val = pd.concat([df_val, df_val_new], ignore_index=True)
    
    return df_val

# Update the custom_train_eval function with improved checkpoint saving
def custom_train_eval(model, ld_tr, ld_val, opt, sched, writer, epochs, scaler):
    # Skip final training if FINAL_EPOCHS is 0
    if epochs <= 0:
        print("Skipping final training as FINAL_EPOCHS is set to 0")
        return
    
    df_train = pd.DataFrame(columns=[
        "epoch", "loss_classifier", "loss_box_reg", "loss_objectness",
        "loss_rpn_box_reg", "loss_total", "mAP", "precision", "recall"
    ])
    
    df_val = pd.DataFrame(columns=[
        "epoch", "loss_classifier", "loss_box_reg", "loss_objectness",
        "loss_rpn_box_reg", "loss_total", "mAP", "precision", "recall"
    ])
        
    for ep in range(1, epochs+1):
        train_one_epoch(model, opt, ld_tr, DEVICE, ep, writer, scaler)
        sched.step()
        
        # Create weights directory if it doesn't exist
        os.makedirs("weights", exist_ok=True)
        # Save model less frequently to save disk space
        # Save at regular intervals and always save the final model
        if ep == 1:
            df_train = val_results(model, ld_tr, DEVICE, writer, ep, df_train, type_split='train')
            df_val = val_results(model, ld_val, DEVICE, writer, ep, df_val, type_split='val')
            folder_train  = "runs_RCNN/final/train_results.csv"
            folder_val    = "runs_RCNN/final/val_results.csv"
            df_train.to_csv(folder_train, index=False)
            df_val.to_csv(folder_val, index=False)
        elif ep % 10 == 0 or ep == epochs:  # Save every 5th epoch for shorter training
            #save the DataFrames to CSV files
            #validate(model, ld_val, DEVICE, writer, ep)
            df_train = val_results(model, ld_tr, DEVICE, writer, ep, df_train, type_split='train')
            df_val = val_results(model, ld_val, DEVICE, writer, ep, df_val, type_split='val')
            folder_train  = "runs_RCNN/final/train_results.csv"
            folder_val    = "runs_RCNN/final/val_results.csv"
            df_train.to_csv(folder_train, index=False)
            df_val.to_csv(folder_val, index=False)
            torch.save(model.base_model.state_dict(), f"weights/fasterrcnn_final_epoch_{ep}.pth")
            print(f"Saved checkpoint epoch {ep}")
        else:
            print(f"Completed epoch {ep} (checkpoint not saved)")
        
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # Create directories for saving outputs
    os.makedirs("weights", exist_ok=True)
    os.makedirs("runs_RCNN", exist_ok=True)

    try:
        # Report compilation availability with enhanced error handling
        if has_compile:
            print("PyTorch compilation available - will use torch.compile for speedup")
        else:
            print("Running without PyTorch compilation")
        
        study = joblib.load("optuna_radiograph_study.pkl")
        
        print('Best params loaded:', study.best_trial.params)
        best = study.best_trial.params
        
        

        # If FINAL_EPOCHS > 0, run final training
        if FINAL_EPOCHS > 0:
            print(f"\nStarting final training with {FINAL_EPOCHS} epochs using best hyperparameters")
            
            writer = SummaryWriter(log_dir='runs_RCNN/final')
            
            # Create DataLoaders with NO multiprocessing
            ds_tr = YOLODataset(TRAIN_DIR, transforms=train_transform)
            ld_tr = DataLoader(
                ds_tr, 
                batch_size=BATCH, 
                shuffle=True,
                num_workers=0,  # Force no multiprocessing
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            ds_val = YOLODataset(VAL_DIR, transforms=None)
            ld_val = DataLoader(
                ds_val, 
                batch_size=BATCH, 
                shuffle=False,
                num_workers=0,  # Force no multiprocessing
                pin_memory=True,
                collate_fn=collate_fn
            )

            base = make_model(best['w_box'], best['w_rpn']).to(DEVICE)
            
            # Apply torch.compile with enhanced error handling
            if has_compile:
                try:
                    print("Applying PyTorch compilation for speedup")
                    base = torch.compile(base)
                except Exception as e:
                    print(f"Warning: Failed to apply torch.compile: {e}")
                    print("Continuing without PyTorch compilation")
                
            model = ModelWithCustomForward(
                base,
                best['w_cls'],
                best['w_box'],
                best['w_obj'],
                best['w_rpn']
            ) 
            
            scaler = GradScaler()
            opt = torch.optim.Adam(base.parameters(), lr=best['lr'])
            
            # Use CosineAnnealingWarmRestarts for final training
            sched = CosineAnnealingWarmRestarts(
                opt,
                T_0=10,      # Initial cycle length
                T_mult=2,    # Multiply cycle length by this factor after each restart  
                eta_min=1e-6 # Minimum learning rate
            )

            custom_train_eval(model, ld_tr, ld_val, opt, sched, writer, FINAL_EPOCHS, scaler)
            writer.close()
            print('Final training complete')
        else:
            print("Skipping final training as FINAL_EPOCHS is set to 0")
        
        print("\nEntire process completed successfully")
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Don't try to set multiprocessing start method at all
    main()