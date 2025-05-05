
"""
Train Faster R-CNN using pre-tuned hyperparameters and pretrained weights.
Loads:
- Hyperparameters from Optuna study file
- Best model weights from specified path
Tracks and saves performance metrics per epoch to CSV
"""

import os
import random
import csv
from pathlib import Path
import sys
from collections import defaultdict

import torch
# Enable cudnn autotune for optimal conv algorithms
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import joblib  # for loading Optuna study
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms as T
import torchvision.transforms.functional as F
import numpy as np

# ----------------------
# Configuration
# ----------------------
TRAIN_DIR       = Path("Data/split_data/train")
VAL_DIR         = Path("Data/split_data/val")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 16
NUM_WORKERS     = 0    # for deterministic loading
NUM_EPOCHS      = 80
MIN_SIZE        = 416
MAX_SIZE        = 416
ACCUM_STEPS     = 2
FREEZE_BACKBONE = True

# Paths for loading
WEIGHTS_PATH    = r"C:\Users\Soren\Documents\amld2025_fracAtlas\weights\fasterrcnn_best.pth"
STUDY_PATH      = "optuna_radiograph_study.pkl"
METRICS_CSV     = "training_metrics.csv"  # CSV file to save metrics

# Default hyperparameters (fallbacks)
DEFAULT_LR       = 1e-4
DEFAULT_DECAY    = 1e-4
DEFAULT_W_CLS    = 1.0
DEFAULT_W_BOX    = 1.0
DEFAULT_W_OBJ    = 1.0
DEFAULT_W_RPN    = 1.0

# IoU thresholds for mAP calculation
IOU_THRESHOLDS = [0.5]  # for mAP50
IOU_RANGE = np.linspace(0.5, 0.95, 10)  # for mAP50-95

# ----------------------
# Utilities
# ----------------------
def collate_fn(batch):
    return tuple(zip(*batch))

def train_transform(image, target):
    # Random horizontal flip
    if random.random() < 0.5:
        _, H, W = image.shape
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0,2]] = W - boxes[:, [2,0]]
        target["boxes"] = boxes

    # Small random rotation
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        pil = T.ToPILImage()(image)
        image = T.ToTensor()(pil.rotate(angle, expand=False))

    # Contrast jitter
    if random.random() < 0.5:
        pil = T.ToPILImage()(image)
        image = T.ToTensor()(T.ColorJitter(brightness=0.2, contrast=0.3)(pil))

    # Gamma adjustment
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        image = torch.pow(image, gamma)

    return image, target

class YOLODataset(Dataset):
    def __init__(self, base_dir, transforms=None):
        base_dir = Path(base_dir)
        self.images = sorted((base_dir/"images").glob("*.[jp][pn]g"))
        self.labels = base_dir/"labels"
        if not self.images:
            raise RuntimeError(f"No images found in {base_dir/'images'}")
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = read_image(str(img_path)).float() / 255.0
        _, H, W = img.shape

        txt = self.labels/f"{img_path.stem}.txt"
        boxes, labels = [], []
        if txt.exists():
            for line in txt.read_text().splitlines():
                c, xc, yc, w0, h0 = map(float, line.split())
                xc, yc, w0, h0 = xc*W, yc*H, w0*W, h0*H
                x0, y0 = xc - w0/2, yc - h0/2
                boxes.append([x0, y0, x0+w0, y0+h0])
                labels.append(int(c))
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

def make_model(freeze_backbone=FREEZE_BACKBONE):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=0.4,
        box_nms_thresh=0.5,
        box_detections_per_img=10,
    )

    if freeze_backbone:
        for p in model.backbone.parameters(): p.requires_grad = False
        print("Backbone frozen.")

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

class ModelWithCustomForward(torch.nn.Module):
    def __init__(self, base, w_cls, w_box, w_obj, w_rpn):
        super().__init__()
        self.base = base
        self.w_cls, self.w_box, self.w_obj, self.w_rpn = w_cls, w_box, w_obj, w_rpn

    def forward(self, images, targets=None):
        if targets is not None:
            # Training or validation with loss calculation
            loss_dict = self.base(images, targets)
            
            # Check if loss_dict is a proper dictionary
            if not isinstance(loss_dict, dict):
                print(f"Warning: loss_dict is not a dictionary: {type(loss_dict)}")
                return loss_dict  # Return as is if not a dictionary
                
            # Calculate weighted loss
            try:
                total = (
                    self.w_cls * loss_dict['loss_classifier']
                    + self.w_box * loss_dict['loss_box_reg']
                    + self.w_obj * loss_dict['loss_objectness']
                    + self.w_rpn * loss_dict['loss_rpn_box_reg']
                )
                loss_dict['loss_total'] = total
                return loss_dict
            except KeyError as e:
                print(f"KeyError in forward: {e}")
                print(f"Available keys: {loss_dict.keys()}")
                # Add a fallback total loss calculation
                if loss_dict:
                    loss_dict['loss_total'] = sum(v for v in loss_dict.values())
                return loss_dict
        else:
            # Inference mode
            return self.base(images)

def train_one_epoch(model, optimizer, loader, device, epoch, writer, scaler):
    model.train()
    optimizer.zero_grad()
    for step, (imgs, tgts) in enumerate(tqdm(loader, desc=f"Train E{epoch}")):
        images = [img.to(device) for img in imgs]
        targets = [{k:v.to(device) for k,v in t.items()} for t in tgts]
        with autocast():
            loss_dict = model(images, targets)
            loss = loss_dict['loss_total'] / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step+1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        writer.add_scalar('train/total_loss', loss_dict['loss_total'].item(), epoch*len(loader)+step)

# ---------------------------------
# Object Detection Metric Functions
# ---------------------------------
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

def calculate_metrics(all_detections, all_targets, iou_threshold=0.5):
    """
    Calculate AP, Precision, and Recall
    
    Arguments:
        all_detections: list of detection dictionaries (boxes, scores, labels)
        all_targets: list of target dictionaries (boxes, labels)
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        ap: Average Precision
        precision: Precision
        recall: Recall
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
            
            # Skip background class
            if pred_label == 0:
                continue
                
            scores.append(pred_scores[idx].item())
            
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
        return 0.0, 0.0, 0.0
    
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
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    
    # Ensure precision is non-increasing for each recall value
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP
    ap = compute_ap(recalls, precisions)
    
    # Return AP, precision, and recall
    # Use last precision and recall values if available, otherwise 0
    if len(precisions) > 2:
        return ap, precisions[-2], recalls[-2]
    else:
        return ap, 0.0, 0.0
    
def evaluate_model(model, loader, device):
    all_detections = []
    all_targets    = []
    loss_sums      = defaultdict(float)
    count          = 0

    with torch.no_grad():
        for imgs, tgts in tqdm(loader, desc="Evaluating"):
            images  = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            # ----- compute losses -----
            model.base.train()                  # force training mode so base(...) returns losses
            loss_dict = model(images, targets)  # wrapper.forward goes to the loss branch
            for k, v in loss_dict.items():
                loss_sums[k] += v.item()
            count += 1

            # ----- compute detections -----
            model.base.eval()                   # switch to eval so base(...) returns boxes/scores
            preds = model(images)               # inference branch of your wrapper

            for i, (pred, target) in enumerate(zip(preds, targets)):
                try:
                    # Move everything to CPU (and detach preds from the graph)
                    all_detections.append({
                        'boxes':  pred['boxes'].detach().cpu(),
                        'scores': pred['scores'].detach().cpu(),
                        'labels': pred['labels'].detach().cpu(),
                    })
                    all_targets.append({
                        'boxes':  target['boxes'].cpu(),
                        'labels': target['labels'].cpu(),
                    })

                except TypeError as e:
                    # Debug output for unexpected prediction format
                    print(f"Error processing prediction {i}: {e}")
                    print(f"Prediction type: {type(pred)}")
                    print(f"Prediction content: {pred}")

                    # Try to handle the case where pred might be a list instead of dict
                    if isinstance(pred, list) and len(pred) >= 3:
                        print("Treating prediction as a list")
                        all_detections.append({
                            'boxes':  pred[0].detach().cpu(),
                            'scores': pred[1].detach().cpu(),
                            'labels': pred[2].detach().cpu(),
                        })
                        all_targets.append({
                            'boxes':  target['boxes'].cpu(),
                            'labels': target['labels'].cpu(),
                        })
                    else:
                        # Skip this prediction if we can't handle the format
                        print("Skipping this prediction due to format issues")
                        continue

    # Calculate metrics at IoU=0.5
    ap50, precision, recall = calculate_metrics(all_detections, all_targets, iou_threshold=0.5)

    # Calculate mAP@50-95
    aps = []
    for iou in IOU_RANGE:
        ap, _, _ = calculate_metrics(all_detections, all_targets, iou_threshold=iou)
        aps.append(ap)
    map50_95 = np.mean(aps)

    # Calculate average losses
    avg_losses = {k: v/count for k, v in loss_sums.items()}

    return {
        'mAP50': ap50,
        'mAP50-95': map50_95,
        'precision': precision,
        'recall': recall,
        **avg_losses
    }


    
def validate(model, loader, device, writer, epoch):
    """
    Validate model and compute metrics
    """
    print(f"Starting validation for epoch {epoch}...")
    model.eval()
    metrics = evaluate_model(model, loader, device)
    
    # Log to tensorboard
    for name, value in metrics.items():
        writer.add_scalar(f'val/{name}', value, epoch)
    
    print(f"[E{epoch}] Validation metrics:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Loss: {metrics['loss_total']:.4f}")
    
    return metrics

def save_metrics_to_csv(metrics_list, filepath):
    """
    Save metrics history to CSV file
    """
    if not metrics_list:
        return
    
    fieldnames = ['epoch'] + list(metrics_list[0].keys())
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, metrics in enumerate(metrics_list, 1):
            row = {'epoch': epoch, **metrics}
            writer.writerow(row)
    
    print(f"Metrics saved to {filepath}")

def main():
    # Load study and best params
    study = joblib.load(STUDY_PATH)
    best = study.best_trial.params
    print("Using best hyperparameters from study:")
    for k, v in best.items(): print(f"  {k}: {v}")

    # Local hyperparams
    lr           = best.get('lr', DEFAULT_LR)
    decay        = best.get('weight_decay', DEFAULT_DECAY)
    w_cls, w_box = best.get('w_cls', DEFAULT_W_CLS), best.get('w_box', DEFAULT_W_BOX)
    w_obj, w_rpn = best.get('w_obj', DEFAULT_W_OBJ), best.get('w_rpn', DEFAULT_W_RPN)

    # Data loaders
    ds_tr  = YOLODataset(TRAIN_DIR, transforms=train_transform)
    ds_val = YOLODataset(VAL_DIR, transforms=None)
    ld_tr  = DataLoader(ds_tr, 
                        batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_fn)
    ld_val = DataLoader(ds_val,
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_fn)

    # Model and weights
    base = make_model().to(DEVICE)
    base.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("Loaded weights from", WEIGHTS_PATH)

    model = ModelWithCustomForward(base, w_cls, w_box, w_obj, w_rpn).to(DEVICE)
    scaler = GradScaler()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base.parameters()),
        lr=lr, weight_decay=decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    writer = SummaryWriter(log_dir="runs/fasterrcnn_loaded_params")

    best_loss = float('inf')
    os.makedirs("weights", exist_ok=True)
    
    # List to store metrics for each epoch
    metrics_history = []

    for epoch in range(1, NUM_EPOCHS+1):
        train_one_epoch(model, optimizer, ld_tr, DEVICE, epoch, writer, scaler)
        scheduler.step()
        
        # Validate and compute metrics
        metrics = validate(model, ld_val, DEVICE, writer, epoch)
        metrics_history.append(metrics)
        
        # Save metrics to CSV after each epoch
        save_metrics_to_csv(metrics_history, METRICS_CSV)
        
        loss = metrics['loss_total']
        if loss < best_loss:
            best_loss = loss
            torch.save(base.state_dict(), "weights/fasterrcnn_best_loaded.pth")
            print(f"New best model saved (loss {best_loss:.4f})")

        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            torch.save(base.state_dict(), f"weights/fasterrcnn_epoch_{epoch}_loaded.pth")
            print(f"Checkpoint saved: epoch {epoch}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    writer.close()
    print("Training complete.")
    print(f"All metrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()