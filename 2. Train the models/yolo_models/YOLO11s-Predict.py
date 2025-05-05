import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import json

# Set paths
cwd = os.getcwd()
input_dir = os.path.join(cwd, "Data", "split_data")
test_images_dir = os.path.join(input_dir, "test", "images")
test_labels_dir = os.path.join(input_dir, "test", "labels")  # Path to ground truth labels
results_dir = os.path.join(cwd, "inference_results")
os.makedirs(results_dir, exist_ok=True)

# Load your trained model
WEIGHTS = r"C:\Users\Soren\Documents\amld2025_fracAtlas\runs_YOLO\train_results_yolo11s\weights\best.pt"
model = YOLO(WEIGHTS)

# Run validation on the test set to get metrics
val_results = model.val(data=os.path.join(input_dir, 'dataset.yaml'), split='test')

# Store metrics in a dictionary
metrics = {
    "mAP50-95": float(val_results.box.map),
    "mAP50": float(val_results.box.map50),
    "Precision": float(val_results.box.p),
    "Recall": float(val_results.box.r),
    "F1-Score": float(val_results.box.f1)
}

# Print metrics
for key, value in metrics.items():
    print(f"{key}: {value}")

# Save metrics to file
with open(os.path.join(results_dir, 'metrics_results.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# Also save as text file for easy reading
with open(os.path.join(results_dir, 'metrics_results.txt'), 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# Function to read YOLO format labels
def read_yolo_label(label_path, img_width, img_height):
    """
    Read YOLO format label file and convert to [x1, y1, x2, y2] format
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = line.strip().split()
        if len(data) >= 5:  # class x_center y_center width height
            class_id = int(data[0])
            x_center = float(data[1]) * img_width
            y_center = float(data[2]) * img_height
            width = float(data[3]) * img_width
            height = float(data[4]) * img_height
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append((class_id, [x1, y1, x2, y2]))
    
    return boxes

# Get list of test images
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Run inference on test images - now looking for 10 positive predictions
positive_predictions = []
for img_path in test_images:
    results = model.predict(img_path, conf=0.25)
    
    # Check if there's a 'fractured' class detection (class_id=1)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls.item() == 1 and box.conf.item() > 0.5:  # class 1 (fractured) with confidence > 0.5
                positive_predictions.append((img_path, r))
                break
        if len(positive_predictions) >= 10:  # Changed from 5 to 10
            break
    if len(positive_predictions) >= 10:  # Fixed indentation error
        break

# If we couldn't find 10 with high confidence, lower the threshold
if len(positive_predictions) < 10:
    remaining_needed = 10 - len(positive_predictions)
    print(f"Only found {len(positive_predictions)} predictions with confidence > 0.5, lowering threshold to find {remaining_needed} more")
    
    # Get all images that aren't already in positive_predictions
    used_image_paths = [p[0] for p in positive_predictions]
    remaining_images = [img for img in test_images if img not in used_image_paths]
    
    for img_path in remaining_images:
        results = model.predict(img_path, conf=0.2)  # Lower confidence threshold
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls.item() == 1 and box.conf.item() > 0.2:  # Lower threshold
                    positive_predictions.append((img_path, r))
                    break
            if len(positive_predictions) >= 10:
                break
        if len(positive_predictions) >= 10:
            break

print(f"Found {len(positive_predictions)} positive predictions to visualize")

# Save individual positive predictions with both predicted and ground truth boxes
for i, (img_path, result) in enumerate(positive_predictions[:10]):  
    # Read the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_height, img_width = img.shape[:2]
    
    # Create a filename based on the original image name
    img_name = os.path.basename(img_path)
    base_name, ext = os.path.splitext(img_name)
    output_path = os.path.join(results_dir, f"positive_prediction_{i+1}_{base_name}{ext}")
    
    # Get ground truth label path
    label_path = os.path.join(test_labels_dir, f"{base_name}.txt")
    
    # Create a list to keep track of all label positions to avoid overlap
    used_label_positions = []
    
    # Draw ground truth boxes (green)
    gt_boxes = read_yolo_label(label_path, img_width, img_height)
    for class_id, bbox in gt_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Draw ground truth box in green
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Find a good position for the label that doesn't overlap
        label = f"GT: {'fractured' if class_id == 1 else 'non_fractured'}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_width, label_height = label_size
        
        # Try positions: above the box, below the box, inside the box at top
        possible_positions = [
            (x1, y1 - 10),  # Above the box
            (x1, y2 + 20),  # Below the box
            (x1, y1 + 20)   # Inside the box at top
        ]
        
        # Find a position that doesn't overlap with existing labels
        label_pos = None
        for pos in possible_positions:
            px, py = pos
            overlap = False
            
            # Check if this position overlaps with any existing label
            for used_pos in used_label_positions:
                used_x, used_y = used_pos
                # Check for overlap (simple rectangle intersection)
                if (px < used_x + label_width and px + label_width > used_x and
                    py - label_height < used_y and py > used_y - label_height):
                    overlap = True
                    break
            
            if not overlap:
                label_pos = pos
                used_label_positions.append(pos)
                break
        
        # If all positions overlap, offset from the first choice
        if label_pos is None:
            label_pos = (x1, y1 - 10 - len(used_label_positions) * 20)  # Stack vertically
            used_label_positions.append(label_pos)
        
        # Draw the label at the selected position
        cv2.putText(img_rgb, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw predicted boxes (blue)
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(box.cls.item())
        conf = box.conf.item()
        
        # Draw prediction box in blue
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add prediction label with similar overlap avoidance
        label = f"Pred: {'fractured' if class_id == 1 else 'non_fractured'} ({conf:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_width, label_height = label_size
        
        # Try different positions
        possible_positions = [
            (x1, y1 - 10),  # Above the box
            (x1, y2 + 20),  # Below the box
            (x1, y1 + 20)   # Inside the box at top
        ]
        
        # Find a position that doesn't overlap with existing labels
        label_pos = None
        for pos in possible_positions:
            px, py = pos
            overlap = False
            
            # Check if this position overlaps with any existing label
            for used_pos in used_label_positions:
                used_x, used_y = used_pos
                # Check for overlap
                if (px < used_x + label_width and px + label_width > used_x and
                    py - label_height < used_y and py > used_y - label_height):
                    overlap = True
                    break
            
            if not overlap:
                label_pos = pos
                used_label_positions.append(pos)
                break
        
        # If all positions overlap, offset from the first choice
        if label_pos is None:
            label_pos = (x1, y1 - 10 - len(used_label_positions) * 20)  # Stack vertically
            used_label_positions.append(label_pos)
        
        # Draw the label at the selected position
        cv2.putText(img_rgb, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add legend
    cv2.putText(img_rgb, "Green: Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img_rgb, "Blue: Prediction", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the image
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving with cv2
    cv2.imwrite(output_path, img_bgr)
    print(f"Saved prediction {i+1} to {output_path}")
    

# Create the combined figure - now adjusted for 10 images in a 2x5 grid
num_rows = 2
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

for i, (img_path, result) in enumerate(positive_predictions[:10]):
    # Get the saved image path
    img_name = os.path.basename(img_path)
    base_name, ext = os.path.splitext(img_name)
    saved_img_path = os.path.join(results_dir, f"positive_prediction_{i+1}_{base_name}{ext}")
    
    # Read the saved image (which already has bounding boxes)
    img = cv2.imread(saved_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    axes[i].imshow(img)
    axes[i].set_title(f"Prediction {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'combined_positive_predictions.png'))

# Fix: Save results directly from val_results attributes
try:
    # Get available attributes from val_results
    result_attrs = dir(val_results.box)
    serializable_results = {}
    
    # Get serializable attributes
    for attr in result_attrs:
        if attr.startswith('__'):
            continue
        
        value = getattr(val_results.box, attr)
        
        if hasattr(value, 'tolist'):
            try:
                serializable_results[attr] = value.tolist()
            except:
                serializable_results[attr] = str(value)
        elif isinstance(value, (int, float, str, bool)):
            serializable_results[attr] = value
        else:
            serializable_results[attr] = str(value)
    
    # Save to file
    with open(os.path.join(results_dir, 'full_validation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
        
except Exception as e:
    print(f"Could not save full validation results: {e}")

# Fix: Try to save per-class metrics with safer access
try:
    class_names = ['non_fractured', 'fractured']
    per_class_metrics = {}
    
    # Get overall metrics
    per_class_metrics["overall"] = {
        "mAP50-95": float(val_results.box.map),
        "mAP50": float(val_results.box.map50),
        "Precision": float(val_results.box.p),
        "Recall": float(val_results.box.r),
        "F1-Score": float(val_results.box.f1)
    }
    
    # Try to get per-class metrics if available
    if hasattr(val_results.box, 'ap_class_index'):
        class_indices = val_results.box.ap_class_index.tolist() if hasattr(val_results.box.ap_class_index, 'tolist') else val_results.box.ap_class_index
        
        for i, idx in enumerate(class_indices):
            class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            
            # Get metrics for this class
            per_class_metrics[class_name] = {}
            
            # AP per class
            if hasattr(val_results.box, 'ap_class') and i < len(val_results.box.ap_class):
                ap = val_results.box.ap_class[i]
                per_class_metrics[class_name]["AP"] = float(ap) if hasattr(ap, 'item') else float(ap)
            
            # Add other per-class metrics if available
            for metric in ['precision', 'recall', 'f1']:
                metric_attr = f'{metric}_per_class'
                if hasattr(val_results.box, metric_attr) and i < len(getattr(val_results.box, metric_attr)):
                    value = getattr(val_results.box, metric_attr)[i]
                    per_class_metrics[class_name][metric.capitalize()] = float(value) if hasattr(value, 'item') else float(value)
    
    # Save to file
    with open(os.path.join(results_dir, 'per_class_metrics.json'), 'w') as f:
        json.dump(per_class_metrics, f, indent=4)
    
except Exception as e:
    print(f"Could not save per-class metrics: {e}")

print("Evaluation complete. Results saved to:", results_dir)