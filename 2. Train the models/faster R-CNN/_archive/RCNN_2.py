# Script to load YOLO-format data and train Faster R-CNN with speed optimizations
"""
SOE

Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train & validate a Faster R-CNN ResNet50 FPN model.
Includes TensorBoard logging, Optuna hyperparameter tuning, and speed optimizations.
"""

import os
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
INITIAL_BATCH = 4  # Start with smaller batch size
MAX_BATCH = 16     # Gradually increase to this batch size
EPOCHS = 10
WARMUP_EPOCHS = 3  # Epochs for batch and LR warmup
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

# Batch size scheduler for batch warmup
def get_batch_size_for_epoch(initial_batch, max_batch, epochs, warmup_epochs):
    """Return batch sizes for each epoch with warmup"""
    batch_sizes = []
    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            # Linear increase from initial to max batch size
            batch_size = initial_batch + (max_batch - initial_batch) * (epoch / warmup_epochs)
            batch_size = int(batch_size)
        else:
            batch_size = max_batch
        batch_sizes.append(batch_size)
    return batch_sizes

class GradualBatchSizeDataLoader:
    """Wrapper around DataLoader that changes batch size gradually"""
    def __init__(self, dataset, initial_batch, max_batch, epochs, warmup_epochs, **kwargs):
        self.dataset = dataset
        self.batch_sizes = get_batch_size_for_epoch(initial_batch, max_batch, epochs, warmup_epochs)
        self.current_epoch = 0
        self.kwargs = kwargs
        # Initial loader
        self.loader = DataLoader(
            dataset, 
            batch_size=self.batch_sizes[0],
            **kwargs
        )
        
    def set_epoch(self, epoch):
        """Update the batch size based on the current epoch"""
        self.current_epoch = epoch - 1  # 0-indexed
        batch_size = self.batch_sizes[self.current_epoch]
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            **self.kwargs
        )
        return self.loader
        
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)

# Learning rate scheduler with warmup
def create_warmup_scheduler(optimizer, warmup_epochs, total_epochs, base_lr):
    """Create a learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def make_model(w_box, w_rpn):
    """Faster R-CNN V2 with COCO-V2 weights and 2-class head."""
    # Use the standard model - DO NOT modify anchor generator which causes shape mismatch
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=0.05,  # Lower detection threshold
        rpn_post_nms_top_n_train=2000,  
        rpn_post_nms_top_n_test=1000,
    )
    
    # Only modify the predictor for 2 classes
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
            
        # Log current batch size and learning rate
        if hasattr(loader, 'batch_size'):
            writer.add_scalar('train/batch_size', loader.batch_size, gs)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], gs)

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

def objective(trial):
    # hyperparams
    lr    = trial.suggest_float('lr',    1e-5, 1e-3, log=True)
    w_cls = trial.suggest_float('w_cls', 0.5, 5.0)
    w_box = trial.suggest_float('w_box', 0.5, 5.0)
    w_obj = trial.suggest_float('w_obj', 0.5, 5.0)
    w_rpn = trial.suggest_float('w_rpn', 0.5, 5.0)

    writer = SummaryWriter(log_dir=f"runs_RCNN/trial_{trial.number}")

    # Create dataset
    ds_tr = YOLODataset(TRAIN_DIR, transforms=train_transform)
    ds_val = YOLODataset(VAL_DIR, transforms=None)
    
    # Create adaptive batch size loader
    adaptive_loader = GradualBatchSizeDataLoader(
        ds_tr,
        initial_batch=INITIAL_BATCH,
        max_batch=MAX_BATCH, 
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Validation loader with fixed batch size
    val_loader = DataLoader(
        ds_val, 
        batch_size=INITIAL_BATCH,  # Keep small for validation to save memory
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Create model with custom weights
    base = make_model(w_box, w_rpn).to(DEVICE)
    model = ModelWithCustomForward(base, w_cls, w_box, w_obj, w_rpn)
    
    # Setup optimizer and LR scheduler with warmup
    scaler = GradScaler()
    optimizer = torch.optim.Adam(base.parameters(), lr=lr)
    warmup_scheduler = create_warmup_scheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        base_lr=lr
    )

    # Training loop for hyperparameter optimization
    for ep in range(1, 4):  # Only 3 epochs for hyperparam tuning
        # Update batch size for current epoch
        train_loader = adaptive_loader.set_epoch(ep)
        
        # Train for this epoch
        train_one_epoch(model, optimizer, train_loader, DEVICE, ep, writer, scaler)
        
        # Step the learning rate scheduler
        warmup_scheduler.step()
        
        # Validate
        avg = validate(model, val_loader, DEVICE, writer, ep)
        
        # Report to Optuna
        trial.report(avg['loss_total'], ep)
        if trial.should_prune():
            writer.close()
            raise TrialPruned()
            
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    return avg['loss_total']

def custom_train_eval(model, train_dataset, val_dataset, optimizer, writer, epochs, warmup_epochs, scaler):
    """Custom training function with progressive batch size and learning rate warmup"""
    
    # Create adaptive batch size loader
    adaptive_loader = GradualBatchSizeDataLoader(
        train_dataset,
        initial_batch=INITIAL_BATCH,
        max_batch=MAX_BATCH, 
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        shuffle=True,
        num_workers=0, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Validation loader with fixed batch size
    val_loader = DataLoader(
        val_dataset, 
        batch_size=INITIAL_BATCH,  # Keep small for validation
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create warmup scheduler
    warmup_scheduler = create_warmup_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=optimizer.param_groups[0]['lr']
    )
    
    # Training loop
    for ep in range(1, epochs + 1):
        # Update batch size for current epoch
        train_loader = adaptive_loader.set_epoch(ep)
        current_batch_size = adaptive_loader.batch_sizes[ep-1]
        
        # Train for this epoch
        train_one_epoch(model, optimizer, train_loader, DEVICE, ep, writer, scaler)
        
        # Step the learning rate scheduler
        warmup_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validate
        validate(model, val_loader, DEVICE, writer, ep)
        
        # Save checkpoint
        os.makedirs("weights", exist_ok=True)
        torch.save({
            'epoch': ep,
            'model_state_dict': model.base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': warmup_scheduler.state_dict(),
            'batch_size': current_batch_size,
        }, f"weights/fasterrcnn_epoch_{ep}.pth")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Epoch {ep} completed with batch_size={current_batch_size}, lr={current_lr:.6f}")

def main():
    # Create directories for saving outputs
    os.makedirs("weights", exist_ok=True)
    os.makedirs("runs_RCNN", exist_ok=True)

    try:
        # Check for PyTorch 2.0+ compilation
        has_compile = hasattr(torch, 'compile')
        if has_compile:
            print("PyTorch compilation available - will use torch.compile for speedup")
        
        # Hyperparameter optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2)
        print('Best params:', study.best_trial.params)
        best = study.best_trial.params

        # Setup for final training
        writer = SummaryWriter(log_dir='runs_RCNN/final')
        
        # Create datasets
        train_dataset = YOLODataset(TRAIN_DIR, transforms=train_transform)
        val_dataset = YOLODataset(VAL_DIR, transforms=None)
        
        # Create model with best parameters
        base_model = make_model(best['w_box'], best['w_rpn']).to(DEVICE)
        
        # Apply compilation if available
        if has_compile:
            base_model = torch.compile(base_model)
            
        # Wrap with loss weighting
        model = ModelWithCustomForward(
            base_model,
            best['w_cls'],
            best['w_box'],
            best['w_obj'],
            best['w_rpn']
        )
        
        # Setup optimizer
        scaler = GradScaler()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=best['lr'])
        
        # Run full training with adaptive batch size and LR
        custom_train_eval(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            writer=writer,
            epochs=EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            scaler=scaler
        )
        
        writer.close()
        print('Training complete with all optimizations')
        
    except Exception as e:
        print(f"Error during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()