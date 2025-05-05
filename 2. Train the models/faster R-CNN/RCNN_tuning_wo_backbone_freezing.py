

"""
Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train & validate a Faster R-CNN ResNet50 FPN model.
Includes TensorBoard logging, Optuna hyperparameter tuning, and speed optimizations.
"""

import os
import random
from pathlib import Path
import joblib  # Add joblib for saving Optuna studies
import sys

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
BATCH = 8
OPTUNA_TRIALS = 8    # Increased number of trials for better hyperparameter search
TRIAL_EPOCHS = 10      # Number of epochs per trial
FINAL_EPOCHS = 30     
ACCUM_STEPS = 2
MIN_SIZE = 416
MAX_SIZE = 416
CONTINUE_FROM_STUDY = True  # Flag to continue from existing study
STUDY_PATH = "optuna_radiograph_study.pkl"  # Path to the study file

# Collate fn for variable-size targets
def collate_fn(batch):
    return tuple(zip(*batch))

# Enhanced train-time augmentations to improve detection of non-fractures
def train_transform(image, target):
    # Random horizontal flip
    if random.random() < 0.5:
        _, H, W = image.shape
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0,2]] = W - boxes[:, [2,0]]
        target["boxes"] = boxes

    # Random rotation (small angles) - helps with orientation invariance
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        pil_img = T.ToPILImage()(image)
        pil_img = pil_img.rotate(angle, expand=False)
        image = T.ToTensor()(pil_img)
        # Note: boxes would need update for large rotations, but for
        # small angles the effect is minimal so we skip for efficiency

    # Enhanced contrast augmentation for radiographs
    if random.random() < 0.5:
        pil = T.ToPILImage()(image)
        # Increase contrast range for radiographs
        pil = T.ColorJitter(
            brightness=0.2,
            contrast=0.3,  # Increased contrast range
            saturation=0.0,
            hue=0.0
        )(pil)
        image = T.ToTensor()(pil)
        
    # Random gamma adjustment (good for radiographs)
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        image = torch.pow(image, gamma)
    
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

def make_model(w_box, w_rpn):
    """Faster R-CNN V2 with COCO-V2 weights, custom anchors, and 2-class head."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE,  # Use the constant
        max_size=MAX_SIZE,  # Use the constant
        # Improved parameters for radiograph analysis
        box_score_thresh=0.4,  # Increased threshold for higher confidence
        box_nms_thresh=0.45,   # Adjusted NMS threshold
        box_detections_per_img=10,  # Reduced from default 100 to limit false positives
    )
    # Modified anchors for radiograph features
    anchor_gen = AnchorGenerator(
        # Include more varied scales for radiographs
        sizes=((4, 8, 16, 32, 64),)*5,
        # Include more rectangular shapes common in fractures
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)*5
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

def load_previous_study(study_path):
    """Load a previous Optuna study if available."""
    if os.path.exists(study_path):
        try:
            print(f"Loading previous study from {study_path}")
            study = joblib.load(study_path)
            print(f"Loaded study with {len(study.trials)} trials")
            
            # Print best parameters from loaded study
            if study.best_trial:
                print("Best parameters from previous study:")
                for param_name, param_value in study.best_trial.params.items():
                    print(f"  {param_name}: {param_value}")
                    
            return study
        except Exception as e:
            print(f"Error loading study: {e}")
            print("Creating new study instead")
    else:
        print(f"Study file {study_path} not found. Creating new study.")
    
    return None

def objective(trial):
    # Learning rate
    lr    = trial.suggest_float('lr',    5e-6, 8e-4, log=True) 
    # Weight decay for regularization
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    w_cls = trial.suggest_float('w_cls', 1.2, 2.8)   # Weight range for classification
    w_box = trial.suggest_float('w_box', 2.0, 4.0)   # Box regression (localization)
    w_obj = trial.suggest_float('w_obj', 0.8, 2.5)   # Objectness weight to reduce false positives
    w_rpn = trial.suggest_float('w_rpn', 0.8, 2.0)   # Region proposal importance

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
    # Updated Adam optimizer with weight decay parameter
    opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use CosineAnnealingWarmRestarts
    sched = CosineAnnealingWarmRestarts(
        opt,
        T_0=10,      # Initial cycle length
        T_mult=2,    # Multiply cycle length by this factor after each restart
        eta_min=1e-6 # Minimum learning rate
    )

    # Train for TRIAL_EPOCHS epochs
    for ep in range(1, TRIAL_EPOCHS + 1):
        train_one_epoch(model, opt, ld_tr, DEVICE, ep, writer, scaler)
        sched.step()
        avg = validate(model, ld_val, DEVICE, writer, ep)
        trial.report(avg['loss_total'], ep)
        if trial.should_prune():
            writer.close()
            raise TrialPruned()
            
        # Clear CUDA cache after each epoch to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    writer.close()
    return avg['loss_total']

# Update the custom_train_eval function with improved checkpoint saving
def custom_train_eval(model, ld_tr, ld_val, opt, sched, writer, epochs, scaler):
    # Skip final training if FINAL_EPOCHS is 0
    if epochs <= 0:
        print("Skipping final training as FINAL_EPOCHS is set to 0")
        return
        
    best_val_loss = float('inf')
    
    for ep in range(1, epochs+1):
        train_one_epoch(model, opt, ld_tr, DEVICE, ep, writer, scaler)
        sched.step()
        avg_losses = validate(model, ld_val, DEVICE, writer, ep)
        val_loss = avg_losses['loss_total']
        
        # Create weights directory if it doesn't exist
        os.makedirs("weights", exist_ok=True)
        
        # Save model less frequently to save disk space
        # Save at regular intervals, when we get a new best model, and always save the final model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.base_model.state_dict(), f"weights/fasterrcnn_best.pth")
            print(f"Saved new best model with validation loss: {best_val_loss:.6f}")
        
        if ep % 5 == 0 or ep == epochs:  # Save every 5th epoch for shorter training
            torch.save(model.base_model.state_dict(), f"weights/fasterrcnn_final_epoch_{ep}.pth")
            print(f"Saved checkpoint epoch {ep}")
        else:
            print(f"Completed epoch {ep} (checkpoint not saved)")
        
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_inference(model, image_path, confidence_threshold=0.5):
    """Run inference on a single image with improved handling for radiographs"""
    model.base_model.eval()  # Set model to evaluation mode
    
    # Load and preprocess the image
    image = read_image(image_path).to(torch.float32) / 255.0
    image = image.to(DEVICE)
    
    # Test-time augmentation - run multiple inferences with slight modifications
    predictions_list = []
    
    # Original image
    with torch.no_grad():
        predictions_list.append(model([image])[0])
    
    # Horizontally flipped image
    with torch.no_grad():
        flipped = F.hflip(image)
        pred_flipped = model([flipped])[0]
        
        # Correct bounding box coordinates for the flip
        if 'boxes' in pred_flipped and len(pred_flipped['boxes']) > 0:
            boxes = pred_flipped['boxes'].clone()
            boxes[:, [0, 2]] = image.shape[2] - boxes[:, [2, 0]]
            pred_flipped['boxes'] = boxes
            
        predictions_list.append(pred_flipped)
    
    # Merge predictions with weighted box fusion
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    
    for pred in predictions_list:
        if len(pred['boxes']) > 0:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                if score >= confidence_threshold:
                    merged_boxes.append(box.cpu().numpy())
                    merged_scores.append(score.cpu().numpy())
                    merged_labels.append(label.cpu().numpy())
    
    result = {
        'boxes': merged_boxes,
        'scores': merged_scores,
        'labels': merged_labels
    }
    
    return result

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
        
        # Check for existing study
        study = None
        if CONTINUE_FROM_STUDY:
            study = load_previous_study(STUDY_PATH)
            
        if study is None:
            # Create new study with MedianPruner
            print(f"Starting hyperparameter optimization with {OPTUNA_TRIALS} trials, {TRIAL_EPOCHS} epochs per trial")
            study = optuna.create_study(
                direction='minimize',
                pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1)
            )
        else:
            print(f"Continuing optimization with {OPTUNA_TRIALS} additional trials")
            
        # Use OPTUNA_TRIALS for number of trials
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        
        print('Best params:', study.best_trial.params)
        best = study.best_trial.params
        
        # Save study for later analysis
        joblib.dump(study, STUDY_PATH)
        print(f"Study saved to {STUDY_PATH}")
        
        # Create detailed report of all trials
        trial_data = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = trial.params
                value = trial.value
                trial_data.append({
                    "Trial": i+1,
                    "Loss": value,
                    "Learning Rate": params.get('lr'),
                    "Weight Decay": params.get('weight_decay', 'N/A'),  # Add weight decay to report
                    "Cls Weight": params.get('w_cls'),
                    "Box Weight": params.get('w_box'),
                    "Obj Weight": params.get('w_obj'),
                    "RPN Weight": params.get('w_rpn')
                })
        
        # Print trial results in a formatted table with weight decay
        print("\nHyperparameter Optimization Results:")
        print("-" * 95)
        print(f"{'Trial':^5} | {'Loss':^10} | {'LR':^10} | {'Decay':^10} | {'w_cls':^8} | {'w_box':^8} | {'w_obj':^8} | {'w_rpn':^8}")
        print("-" * 95)
        for trial in sorted(trial_data, key=lambda x: x['Loss']):
            print(f"{trial['Trial']:^5} | {trial['Loss']:^10.6f} | {trial['Learning Rate']:^10.6f} | "
                  f"{trial['Weight Decay']:^10.6f} | {trial['Cls Weight']:^8.2f} | {trial['Box Weight']:^8.2f} | "
                  f"{trial['Obj Weight']:^8.2f} | {trial['RPN Weight']:^8.2f}")
        print("-" * 95)

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
            # Use AdamW with the optimal weight decay
            opt = torch.optim.AdamW(
                base.parameters(), 
                lr=best['lr'], 
                weight_decay=best.get('weight_decay', 1e-5)  # Use default if not found
            )
            
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