import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
# ← allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm.auto import tqdm
import optuna

# ----------------------------
# Utility Functions & Classes
# ----------------------------

def yolo_to_xyxy(label_path, img_w, img_h):
    boxes, labels = [], []
    with open(label_path, 'r') as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.split())
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)
    return boxes, labels

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=320):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_files  = sorted(os.listdir(images_dir))
        self.img_size   = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        w0, h0   = image.size

        lbl_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(lbl_path):
            boxes, labels = yolo_to_xyxy(lbl_path, w0, h0)
        else:
            boxes, labels = [], []

        if self.img_size:
            image = image.resize((self.img_size, self.img_size))
            sx, sy = self.img_size / w0, self.img_size / h0
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                boxes[i] = [x1*sx, y1*sy, x2*sx, y2*sy]

        img_tensor = F.to_tensor(image)
        if boxes:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            # ensure correct shape when no objects
            boxes_tensor  = torch.zeros((0,4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),  dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_map(preds, targets):
    metric = MeanAveragePrecision(iou_thresholds=None)
    metric.update(preds, targets)
    res = metric.compute()
    return res['map_50'].item(), res['map'].item()

def get_num_classes(labels_dir):
    max_cls = 0
    for fn in os.listdir(labels_dir):
        for line in open(os.path.join(labels_dir, fn)):
            c = int(line.split()[0])
            max_cls = max(max_cls, c)
    return max_cls + 1  # YOLO classes start at 0

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model

def train_one_epoch(model, optimizer, loader, device, w_cls, w_bbox, w_dfl):
    model.train()
    total_loss = 0.0
    # progress bar over batches
    for images, targets in tqdm(loader, desc="  Train batches", leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss_cls  = loss_dict['loss_classifier'] + loss_dict['loss_objectness']
        loss_bb   = loss_dict['loss_box_reg']    + loss_dict['loss_rpn_box_reg']
        loss_dfl  = loss_dict.get('loss_dfl', 0.0)
        loss      = w_cls*loss_cls + w_bbox*loss_bb + w_dfl*loss_dfl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in tqdm(loader, desc="  Eval batches", leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        for i, out in enumerate(outputs):
            all_preds.append({
                "boxes":  out['boxes'].cpu(),
                "scores": out['scores'].cpu(),
                "labels": out['labels'].cpu()
            })
            all_targets.append({
                "boxes":  targets[i]['boxes'].cpu(),
                "labels": targets[i]['labels'].cpu()
            })
    return compute_map(all_preds, all_targets)

# ----------------------------
# Main & Optuna Integration
# ----------------------------

if __name__ == "__main__":
    # Paths
    train_img_dir = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\train\images"
    train_lbl_dir = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\train\labels"
    val_img_dir   = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\val\images"
    val_lbl_dir   = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\val\labels"
    test_img_dir  = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\test\images"
    test_lbl_dir  = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\YOLO_dataset\test\labels"

    # Config
    IMG_SIZE     = 320
    BATCH_SIZE   = 32
    DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES  = get_num_classes(train_lbl_dir) + 1
    TRIAL_EPOCHS = 2
    FINAL_EPOCHS = 10
    WEIGHT_DECAY = 1e-4

    # Data loaders
    train_ds = YOLODataset(train_img_dir, train_lbl_dir, IMG_SIZE)
    val_ds   = YOLODataset(val_img_dir,   val_lbl_dir,   IMG_SIZE)
    test_ds  = YOLODataset(test_img_dir,  test_lbl_dir,  IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Optuna objective (maximize mAP50-95 → minimize negative)
    def objective(trial):
        lr    = trial.suggest_float('lr',    1e-5, 1e-3, log=True)
        w_cls = trial.suggest_float('w_cls', 0.5,   2.0)
        w_bb  = trial.suggest_float('w_bbox',0.5,   2.0)
        w_dfl = trial.suggest_float('w_dfl', 0.0,   2.0)

        model     = create_model(NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        for _ in range(TRIAL_EPOCHS):
            train_one_epoch(model, optimizer, train_loader, DEVICE, w_cls, w_bb, w_dfl)

        _, map5095 = evaluate(model, val_loader, DEVICE)
        return -map5095

    # Optuna study without TqdmCallback
    study = optuna.create_study(direction='minimize')
    for _ in tqdm(range(20), desc="Optuna trials"):
        # run exactly one trial per iteration
        study.optimize(objective, n_trials=1)
    best_params = study.best_params
    print("Best hyperparameters:", best_params)


    # Final training
    final_model = create_model(NUM_CLASSES).to(DEVICE)
    optimizer   = torch.optim.Adam(
        final_model.parameters(),
        lr=best_params['lr'],
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, FINAL_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{FINAL_EPOCHS}")
        train_loss = train_one_epoch(
            final_model, optimizer, train_loader, DEVICE,
            best_params['w_cls'], best_params['w_bbox'], best_params['w_dfl']
        )
        map50, map5095 = evaluate(final_model, val_loader, DEVICE)
        print(f" Train Loss: {train_loss:.4f} — Val mAP@0.5: {map50:.3f}, mAP@0.5:0.95: {map5095:.3f}")

    # Test evaluation
    print("\nRunning on test set:")
    test_map50, test_map5095 = evaluate(final_model, test_loader, DEVICE)
    print(f"Test mAP@0.5: {test_map50:.3f}, mAP@0.5:0.95: {test_map5095:.3f}")
