"""
Author: Jieyu Lian
Date: April 2025

Description:
This script evaluates a trained YOLOv11 model on the fracture detection dataset.
It performs both:
- Localization evaluation (object detection metrics like mAP, precision, recall)
- Classification evaluation (fractured vs non-fractured based on predicted boxes)

The script reads the dataset configuration from `dataset.yaml`, loads the trained model,
evaluates performance on the test set, and saves the results to CSV files for reporting.

Outputs:
- yolo_localization_metrics.csv: Detection metrics (mAP, precision, recall)
- yolo_classification_metrics.csv: Classification report (precision, recall, F1)

"""


import os
import argparse
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import classification_report
from pathlib import Path

# === Debug current working directory ===
print("🔍 Current Working Directory:", os.getcwd())

# === Argument parser ===
parser = argparse.ArgumentParser(description="Evaluate YOLO model on fracture detection")

parser.add_argument(
    "--model_path",
    type=str,
    default="../2. Train the models/yolo11n.pt",  
)

parser.add_argument(
    "--dataset_yaml",
    type=str,
    default="../Data/split_data/dataset.yaml",  
    help="Path to dataset.yaml file for validation"
)

parser.add_argument(
    "--test_image_dir",
    type=str,
    default="../Data/split_data/test/images",  
    help="Path to test image directory"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="results",
    help="Directory to save evaluation results"
)

args = parser.parse_args()

# === Path check ===
if not os.path.isfile(args.dataset_yaml):
    raise FileNotFoundError(f"❌ dataset.yaml not found: {args.dataset_yaml}")
if not os.path.isfile(args.model_path):
    raise FileNotFoundError(f"❌ model weights not found: {args.model_path}")
if not os.path.isdir(args.test_image_dir):
    raise FileNotFoundError(f"❌ test image dir not found: {args.test_image_dir}")

# === Load YOLO model ===
model = YOLO(args.model_path)

# === Localization metrics ===
print("\n📊 Evaluating localization (object detection)...")
metrics = model.val(data=args.dataset_yaml, split='test', save=False)

loc_metrics = {
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map,
    "Precision": metrics.box.mp,
    "Recall": metrics.box.mr
}

print("\n--- Localization Metrics ---")
for k, v in loc_metrics.items():
    print(f"{k}: {v:.4f}")

# === Save localization metrics ===
os.makedirs(args.output_dir, exist_ok=True)
pd.DataFrame([loc_metrics]).to_csv(os.path.join(args.output_dir, "yolo_localization_metrics.csv"), index=False)

# === Classification estimation ===
from sklearn.metrics import classification_report

y_true = []
y_pred = []

test_images = sorted(Path(args.test_image_dir).glob("*.jpg"))
for image_path in test_images:
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)
    pred_boxes = results[0].boxes
    num_preds = len(pred_boxes)

    # Use heuristic based on filename to determine ground truth label
    is_fractured = "Fractured" in str(image_path)
    true_label = 1 if is_fractured else 0
    pred_label = 1 if num_preds > 0 else 0

    y_true.append(true_label)
    y_pred.append(pred_label)

# === Save classification report ===
report = classification_report(y_true, y_pred, target_names=['non-fractured', 'fractured'], output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(args.output_dir, "yolo_classification_metrics.csv"))

print("\n--- Classification Metrics ---")
print(df_report[['precision', 'recall', 'f1-score', 'support']])

