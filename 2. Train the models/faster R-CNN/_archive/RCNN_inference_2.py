import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
CWD = os.getcwd()
TEST_DIR = os.path.join(CWD, "Data", "split_data", "test")
WEIGHTS = os.path.join(CWD, "weights", "fasterrcnn_final_epoch_20.pth")
OUTPUT_DIR = os.path.join(CWD, "evaluation_results")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH = 0.5  # Detection confidence threshold
BATCH = 16
MIN_SIZE = 320
MAX_SIZE = 320

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DATASET DEFINITION ──────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))

class YOLODataset(Dataset):
    """YOLO-format dataset loader for test images."""
    def __init__(self, base_dir):
        base_dir = Path(base_dir)
        self.image_dir = base_dir / "images"
        self.label_dir = base_dir / "labels"
        assert self.image_dir.exists(), f"{self.image_dir} not found"
        assert self.label_dir.exists(), f"{self.label_dir} not found"
        self.image_files = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_image(str(img_path)).to(torch.float32) / 255.0
        _, h, w = image.shape
        # load YOLO-format labels
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                c, xc, yc, wn, hn = map(float, line.split())
                xc, yc, wn, hn = xc * w, yc * h, wn * w, hn * h
                x0, y0 = xc - wn / 2, yc - hn / 2
                x1, y1 = xc + wn / 2, yc + hn / 2
                boxes.append([x0, y0, x1, y1])
                labels.append(int(c))
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": idx, "file_path": str(img_path)}
        return image, target

# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
def make_model():
    """Recreate Faster R-CNN V2 with custom anchors and 2-class head, matching the training architecture."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,  # Don't load pretrained weights
        min_size=MIN_SIZE,
        max_size=MAX_SIZE
    )
    
    # Custom anchors matching the training configuration
    anchor_gen = AnchorGenerator(
        sizes=((8,16,32),)*5,
        aspect_ratios=((0.5,1.0,2.0),)*5
    )
    model.rpn.anchor_generator = anchor_gen
    
    # Set up RPN head
    in_channels = model.backbone.out_channels
    num_anchors = anchor_gen.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(in_channels, num_anchors)
    
    # 2-class predictor
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    
    return model

# ─── EVALUATION UTILITIES ────────────────────────────────────────────────────
def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """Compute COCO mAP@IoU=0.5 and class precision."""
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], class_metrics=True)
    model.eval()
    
    # Store predictions and ground truth for each image
    all_predictions = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            inputs = [img.to(device) for img in images]
            outputs = model(inputs)
            preds, gts = [], []
            
            for i, (out, tgt) in enumerate(zip(outputs, targets)):
                # Add file path and other metadata to predictions for later use
                pred_dict = {
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu(),
                }
                
                gt_dict = {
                    'boxes': tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu(),
                }
                
                # Store for metric computation
                preds.append(pred_dict)
                gts.append(gt_dict)
                
                # Store additional metadata for later visualization
                all_predictions.append({
                    'prediction': pred_dict,
                    'ground_truth': gt_dict,
                    'image_path': tgt['file_path'],
                    'image_tensor': images[i]
                })
                
            metric.update(preds, gts)
    
    result = metric.compute()
    return result, all_predictions

def visualize_and_save_positive_detections(predictions, conf_thresh=0.5, max_display=20):
    """Visualize and save only images with positive predictions above the confidence threshold."""
    positive_count = 0
    total_predictions = 0
    
    # First pass: count how many images have positive predictions
    images_with_detections = []
    for pred_info in predictions:
        pred = pred_info['prediction']
        scores = pred['scores']
        keep = scores > conf_thresh
        if torch.any(keep):
            images_with_detections.append(pred_info)
            positive_count += 1
            total_predictions += torch.sum(keep).item()
    
    print(f"\nFound {positive_count} images with positive predictions out of {len(predictions)} total images")
    print(f"Total number of positive detections: {total_predictions}")
    
    # Determine how many images to display
    display_count = min(max_display, len(images_with_detections))
    display_indices = range(display_count)
    
    if display_count == 0:
        print("No positive predictions found above the confidence threshold!")
        return
    
    # Create a multi-image figure to display the images with positive predictions
    fig, axes = plt.subplots(nrows=display_count, ncols=1, figsize=(10, 5*display_count))
    if display_count == 1:
        axes = [axes]  # Make it iterable
        
    for i, idx in enumerate(display_indices):
        pred_info = images_with_detections[idx]
        img = pred_info['image_tensor']
        pred = pred_info['prediction']
        gt = pred_info['ground_truth']
        img_path = pred_info['image_path']
        
        # Get predictions above threshold
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        keep = scores > conf_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Draw bounding boxes on the image
        if len(boxes) > 0:
            drawn_img = draw_bounding_boxes(
                (img * 255).to(torch.uint8),
                boxes,
                colors=['red'] * len(boxes),
                width=3,
                labels=[f"Pred {int(l)}:{s:.2f}" for l, s in zip(labels, scores)]
            )
        else:
            drawn_img = (img * 255).to(torch.uint8)
        
        # Draw ground truth boxes if they exist
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        if len(gt_boxes) > 0:
            drawn_img = draw_bounding_boxes(
                drawn_img,
                gt_boxes,
                colors=['green'] * len(gt_boxes),
                width=2,
                labels=[f"GT {int(l)}" for l in gt_labels]
            )
            
        # Display the image
        axes[i].imshow(drawn_img.permute(1, 2, 0).numpy())
        axes[i].set_title(f"File: {Path(img_path).name}\nPred count: {len(boxes)}, GT count: {len(gt_boxes)}")
        axes[i].axis('off')
        
        # Save individual images with predictions
        plt.figure(figsize=(8, 8))
        plt.imshow(drawn_img.permute(1, 2, 0).numpy())
        plt.title(f"File: {Path(img_path).name}\nPred count: {len(boxes)}, GT count: {len(gt_boxes)}")
        plt.axis('off')
        output_path = os.path.join(OUTPUT_DIR, f"pred_{Path(img_path).stem}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "positive_predictions_summary.png"), bbox_inches='tight')
    plt.show()

# Calculate and print detailed statistics on model performance
def print_detailed_metrics(predictions, conf_thresh=0.5):
    """Calculate and print detailed metrics on model performance."""
    # Count true positives, false positives, false negatives at different IoU thresholds
    total_images = len(predictions)
    total_gt_objects = 0
    total_predicted_objects = 0
    
    # Count images with predictions vs images with ground truth
    images_with_predictions = 0
    images_with_gt = 0
    
    # Count correct vs incorrect predictions
    for pred_info in predictions:
        pred = pred_info['prediction']
        gt = pred_info['ground_truth']
        
        # Count ground truth objects
        gt_count = len(gt['boxes'])
        total_gt_objects += gt_count
        if gt_count > 0:
            images_with_gt += 1
        
        # Count predictions above threshold
        scores = pred['scores']
        keep = scores > conf_thresh
        pred_count = torch.sum(keep).item()
        total_predicted_objects += pred_count
        if pred_count > 0:
            images_with_predictions += 1
    
    print("\n===== DETAILED EVALUATION METRICS =====")
    print(f"Total images evaluated: {total_images}")
    print(f"Images with ground truth objects: {images_with_gt} ({images_with_gt/total_images*100:.1f}%)")
    print(f"Images with positive predictions: {images_with_predictions} ({images_with_predictions/total_images*100:.1f}%)")
    print(f"Total ground truth objects: {total_gt_objects}")
    print(f"Total predicted objects: {total_predicted_objects}")
    
    if total_gt_objects > 0:
        print(f"Overall detection rate: {total_predicted_objects/total_gt_objects:.2f}")
    
    # Calculate precision and recall
    if total_predicted_objects > 0 and total_gt_objects > 0:
        # This is just a rough estimate without computing IoU
        print("\nNote: These are general statistics without precise IoU matching.")
        print(f"For accurate precision/recall values, refer to the mAP calculations.")

# ─── MAIN: TEST EVALUATION ONLY ───────────────────────────────────────────────
if __name__ == '__main__':
    # Load model and weights
    model = make_model().to(DEVICE)
    print("Loading weights from:", WEIGHTS)
    checkpoint = torch.load(WEIGHTS, map_location=DEVICE)
    
    try:
        # Try to load with strict=True first
        model.load_state_dict(checkpoint)
        print("Successfully loaded model weights with strict=True")
    except RuntimeError as e:
        print(f"Error with strict loading: {e}")
        print("Attempting to load with strict=False...")
        
        # Try loading with strict=False
        incompatible_keys = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded with strict=False. Missing keys: {len(incompatible_keys.missing_keys)}, Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
    
    # Prepare test loader
    test_ds = YOLODataset(TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    
    print(f"\nEvaluating model on {len(test_ds)} test images...")
    
    # Quantitative evaluation
    metrics, all_predictions = evaluate_model(model, test_loader, DEVICE, iou_threshold=0.5)
    
    # Print main metrics
    print("\n===== MAIN EVALUATION METRICS =====")
    print(f"mAP@0.5: {metrics['map_50']:.4f}")
    
    # Also print values at multiple IoU thresholds if available
    if 'map' in metrics:
        print(f"mAP@[0.5:0.95]: {metrics['map']:.4f}")
        
    for iou_level in [75, 85, 95]:
        key = f'map_{iou_level}'
        if key in metrics:
            print(f"mAP@0.{iou_level}: {metrics[key]:.4f}")
    
    # Check for class metrics
    if 'class_precision' in metrics and metrics['class_precision'].numel() > 0:
        for i, prec in enumerate(metrics['class_precision']):
            print(f"Class {i} precision: {prec.item():.4f}")
            
        for i, rec in enumerate(metrics['class_recall']):
            print(f"Class {i} recall: {rec.item():.4f}")
    else:
        print("Warning: Detailed class metrics not available")
    
    # Print detailed statistics
    print_detailed_metrics(all_predictions, conf_thresh=CONF_THRESH)
    
    # Visualize and save positive predictions only
    print("\nVisualizing and saving images with positive predictions...")
    visualize_and_save_positive_detections(all_predictions, conf_thresh=CONF_THRESH)
    
    print(f"\nEvaluation complete. Results saved to {OUTPUT_DIR}")