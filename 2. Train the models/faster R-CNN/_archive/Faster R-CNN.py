'''
Author: Jieyu Lian  
Date: 18/04/2025  
Description: Full Faster R-CNN training script with save/load and evaluation.  
             Uses local image folder and YOLO-style annotations.
'''

import os
import torch
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Set relative paths ===
fracAtlas = "../Data/FracAtlas"
image_folder = os.path.join(fracAtlas, "images", "all")
annotation_folder = os.path.join(fracAtlas, "Annotations", "YOLO")
csv_path = os.path.join(fracAtlas, "dataset.csv")

# === Load dataset metadata ===
df = pd.read_csv(csv_path)
df = df.rename(columns={"fractured": "label"})

# === Custom Dataset ===
class FracRCNNDataset(Dataset):
    def __init__(self, dataframe, image_folder, annotation_folder, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, "image_id"]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = load_yolo_annotations(image_name, self.annotation_folder)
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        converted_boxes = []
        for box in boxes:
            x_c, y_c, w, h = box
            xmin = (x_c - w / 2) * width
            ymin = (y_c - h / 2) * height
            xmax = (x_c + w / 2) * width
            ymax = (y_c + h / 2) * height
            converted_boxes.append([xmin, ymin, xmax, ymax])

        boxes_tensor = torch.tensor(converted_boxes, dtype=torch.float32)
        labels_tensor = torch.ones((len(boxes_tensor),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# === Helper: Load YOLO annotation files ===
def load_yolo_annotations(image_name, yolo_folder):
    path = os.path.join(yolo_folder, image_name.replace(".jpg", ".txt"))
    boxes = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                box = list(map(float, parts[1:]))
                boxes.append(box)
    return boxes

# === Dataloader Collate ===
def collate_fn(batch):
    return tuple(zip(*batch))

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# === Train/Test Split ===
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=1)
train_dataset = FracRCNNDataset(df_train, image_folder, annotation_folder, transform)
test_dataset = FracRCNNDataset(df_test, image_folder, annotation_folder, transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# === Initialize Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.to(device)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=0.005, momentum=0.9, weight_decay=0.0005)

# === Training Loop ===
num_epochs = 5
print("\n🚀 Starting training loop...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# === Save Model ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fasterrcnn_fracAtlas.pth")
print("✅ Model saved to models/fasterrcnn_fracAtlas.pth")

# === Prediction Visualization ===
def predict_and_plot(model, dataset, idx=0, threshold=0.5):
    model.eval()
    image, target = dataset[idx]
    image = image.to(device)
    with torch.no_grad():
        prediction = model([image])[0]

    image = image.cpu().permute(1, 2, 0).numpy()
    plt.imshow(image)

    for box, score in zip(prediction["boxes"], prediction["scores"]):
        if score > threshold:
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              edgecolor='red', facecolor='none', linewidth=2))
    plt.axis('off')
    plt.title(f"Prediction on test image #{idx}")
    plt.show()

# === Predict on 1 sample ===
predict_and_plot(model, test_dataset, idx=0)
