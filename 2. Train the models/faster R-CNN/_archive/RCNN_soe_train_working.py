"""
SOE

Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train a Faster R-CNN ResNet50 FPN model for 10 epochs.
Images are resized by the model transform to 320×320 to reduce computation.
Designed for use with torchvision's detection APIs.

**Notes:**
- Uses weights enum instead of deprecated pretrained.
- Prints individual loss components and uses 1-based epoch numbering.
- On Windows, set num_workers=0 to avoid multiprocessing spawn errors.
- Custom RPN and ROI thresholds with weighted box-regression losses.
- Switch to Adam optimizer with gradient clipping.
- Logs average magnitude of each loss component per epoch for balancing.
- Measures iteration time and GPU memory usage per iteration.
- Displays a tqdm progress bar for each epoch.
"""

import os
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# target image size (model will resize internally)
TARGET_SIZE = (320, 320)

class YOLODataset(Dataset):
    """
    Dataset for YOLO-format object detection data.
    Images and labels in separate subfolders; labels normalized.
    """
    def __init__(self, base_dir, transforms=None):
        base_dir = Path(base_dir)
        self.image_dir = base_dir / "images"
        self.label_dir = base_dir / "labels"
        assert self.image_dir.exists(), f"{self.image_dir} not found"
        assert self.label_dir.exists(), f"{self.label_dir} not found"
        self.transforms = transforms

        self.image_files = sorted([p for p in self.image_dir.iterdir()
                                   if p.suffix.lower() in ['.jpg','jpeg','.png']])
        print(f"[YOLODataset] Found {len(self.image_files)} images in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = read_image(str(img_path)).to(torch.float32) / 255.0
        _, orig_h, orig_w = image.shape

        # load boxes
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip(): continue
                c, x_c, y_c, w_n, h_n = map(float, line.split())
                x_c *= orig_w; y_c *= orig_h
                w_n *= orig_w; h_n *= orig_h
                x_min = x_c - w_n/2; y_min = y_c - h_n/2
                x_max = x_c + w_n/2; y_max = y_c + h_n/2
                boxes.append([x_min, y_min, x_max, y_max])
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


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    sum_dict = {k:0.0 for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']}
    steps = 0

    # wrap data_loader in tqdm for a progress bar
    loader = tqdm(data_loader, desc=f"Epoch {epoch}", unit="iter")
    for images, targets in loader:
        start_time = time.time()
        images = [img.to(device) for img in images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = (
            loss_dict['loss_classifier']
          + 2.0 * loss_dict['loss_box_reg']
          + loss_dict['loss_objectness']
          + 2.0 * loss_dict['loss_rpn_box_reg']
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        iter_time = time.time() - start_time
        mem_mb = torch.cuda.memory_allocated(device) / 1024**2 if device.type=='cuda' else 0.0

        for k in sum_dict:
            sum_dict[k] += loss_dict[k].item()
        steps += 1

        if steps % print_freq == 0:
            loader.set_postfix({
                'loss': total_loss.item(),
                'time(s)': f"{iter_time:.3f}",
                'mem(MB)': f"{mem_mb:.1f}"
            })

    avg = {k: sum_dict[k]/steps for k in sum_dict}
    print(f"[Epoch {epoch}] Averages -> cls: {avg['loss_classifier']:.4f}, box: {avg['loss_box_reg']:.4f}, obj: {avg['loss_objectness']:.4f}, rpn: {avg['loss_rpn_box_reg']:.4f}")

if __name__ == "__main__":
    train_base = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\split_data\train"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = YOLODataset(train_base)
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )

    anchor_gen = AnchorGenerator(
        sizes=((8,16,32),)*5,
        aspect_ratios=((0.5,1.0,2.0),)*5
    )

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_gen,
        rpn_fg_iou_thresh=0.3,
        rpn_bg_iou_thresh=0.1,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        min_size=320,
        max_size=320,
    )
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(1, 11):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()
        torch.save(model.state_dict(), f"fasterrcnn_epoch_{epoch}.pth")
    print("Training complete.")
