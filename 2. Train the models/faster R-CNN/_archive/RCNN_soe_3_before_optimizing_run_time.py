# Script to load YOLO-format data and train Faster R-CNN with speed optimizations
"""
SOE

Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train & validate a Faster R-CNN ResNet50 FPN model.
Includes TensorBoard logging, Optuna hyperparameter tuning, and speed optimizations.
"""

import os
import random
from pathlib import Path

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
import optuna
from optuna import TrialPruned
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
EPOCHS = 100
ACCUM_STEPS = 2
MIN_SIZE = 320  # Define min_size as a constant
MAX_SIZE = 320  # Define max_size as a constant

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

# In the objective function, replace the scheduler
def objective(trial):
    # hyperparams
    lr    = trial.suggest_float('lr',    1e-5, 1e-3, log=True)
    w_cls = trial.suggest_float('w_cls', 0.5, 5.0)
    w_box = trial.suggest_float('w_box', 0.5, 5.0)
    w_obj = trial.suggest_float('w_obj', 0.5, 5.0)
    w_rpn = trial.suggest_float('w_rpn', 0.5, 5.0)

    writer = SummaryWriter(log_dir=f"runs_RCNN/trial_{trial.number}")

    # Create dataset and DataLoader with NO multiprocessing
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

    base = make_model(w_box, w_rpn).to(DEVICE)
    model = ModelWithCustomForward(base, w_cls, w_box, w_obj, w_rpn)
    scaler = GradScaler()
    opt    = torch.optim.Adam(base.parameters(), lr=lr)
    
    # Replace StepLR with CosineAnnealingWarmRestarts
    sched  = CosineAnnealingWarmRestarts(
        opt,
        T_0=10,      # Initial cycle length
        T_mult=2,    # Multiply cycle length by this factor after each restart
        eta_min=1e-6 # Minimum learning rate
    )

    for ep in range(1, 4):
        train_one_epoch(model, opt, ld_tr, DEVICE, ep, writer, scaler)
        sched.step()
        avg = validate(model, ld_val, DEVICE, writer, ep)
        trial.report(avg['loss_total'], ep)
        if trial.should_prune():
            writer.close()
            raise TrialPruned()

    writer.close()
    return avg['loss_total']


def custom_train_eval(model, ld_tr, ld_val, opt, sched, writer, epochs, scaler):
    for ep in range(1, epochs+1):
        train_one_epoch(model, opt, ld_tr, DEVICE, ep, writer, scaler)
        sched.step()
        validate(model, ld_val, DEVICE, writer, ep)
        
        # Create weights directory if it doesn't exist
        os.makedirs("weights", exist_ok=True)
        
        # Save model less frequently to save disk space
        # Save at regular intervals and always save the final model
        if ep % 10 == 0 or ep == epochs:
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
        # Torch.compile for additional speedup
        has_compile = hasattr(torch, 'compile')
        if has_compile:
            print("PyTorch compilation available - will use torch.compile for speedup")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        print('Best params:', study.best_trial.params)
        best = study.best_trial.params

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
        
        # Apply torch.compile if available
        if has_compile:
            base = torch.compile(base)
            
        model = ModelWithCustomForward(
            base,
            best['w_cls'],
            best['w_box'],
            best['w_obj'],
            best['w_rpn']
        )
        
        scaler = GradScaler()
        opt = torch.optim.Adam(base.parameters(), lr=best['lr'])
        
        # Replace StepLR with CosineAnnealingWarmRestarts
        sched = CosineAnnealingWarmRestarts(
            opt,
            T_0=10,      # Initial cycle length
            T_mult=2,    # Multiply cycle length by this factor after each restart  
            eta_min=1e-6 # Minimum learning rate
        )

        custom_train_eval(model, ld_tr, ld_val, opt, sched, writer, EPOCHS, scaler)
        writer.close()
        print('Training complete')
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()