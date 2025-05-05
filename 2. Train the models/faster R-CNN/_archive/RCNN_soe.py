"""
SOE

Script to load YOLO-format object detection data into a PyTorch Dataset,
visualize samples, and train & validate a Faster R-CNN ResNet50 FPN model.
Includes TensorBoard logging and Optuna hyperparameter tuning.
"""

import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm
import optuna
from optuna import TrialPruned

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

# Constants
TRAIN_DIR = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\split_data\train"
VAL_DIR   = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\split_data\val"
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH     = 8
EPOCHS    = 2

# Collate function for variable-size targets
def collate_fn(batch):
    return tuple(zip(*batch))

class YOLODataset(Dataset):
    """Dataset for YOLO-format object detection data."""
    def __init__(self, base_dir, transforms=None):
        base_dir = Path(base_dir)
        self.image_dir = base_dir / "images"
        self.label_dir = base_dir / "labels"
        assert self.image_dir.exists(), f"{self.image_dir} not found"
        assert self.label_dir.exists(), f"{self.label_dir} not found"
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

        # Load YOLO-format labels
        label_path = self.label_dir / img_path.with_suffix('.txt').name
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                c, xc, yc, wn, hn = map(float, line.split())
                xc *= w; yc *= h
                wn *= w; hn *= h
                x0, y0 = xc - wn/2, yc - hn/2
                x1, y1 = xc + wn/2, yc + hn/2
                boxes.append([x0, y0, x1, y1])
                labels.append(int(c))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def make_model(w_box, w_rpn):
    """Builds Faster R-CNN with custom loss weights."""
    # Anchor generator for 5 FPN levels
    anchor_gen = AnchorGenerator(
        sizes=((8,16,32),)*5,
        aspect_ratios=((0.5,1.0,2.0),)*5
    )
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_gen,
        rpn_fg_iou_thresh=0.3, # default 0.7 focusing on positive samples
        rpn_bg_iou_thresh=0.1, # default 0.3 focusing on negative samples
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        min_size=160,
        max_size=160,
    )
    # Attach loss weights to model so args are used
    model.w_box = w_box
    model.w_rpn = w_rpn
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    return model


# Custom forward method wrapper to ensure loss dictionary is returned in both train and eval modes
class ModelWithCustomForward(torch.nn.Module):
    def __init__(self, base_model, w_box, w_rpn):
        super().__init__()
        self.base_model = base_model
        self.w_box = w_box
        self.w_rpn = w_rpn
        
    def forward(self, images, targets=None):
        # Always set train mode temporarily when targets are provided (for loss computation)
        if targets is not None:
            training = self.base_model.training
            if not training:
                self.base_model.train()
                
            loss_dict = self.base_model(images, targets)
            
            # Restore original mode
            if not training:
                self.base_model.eval()
                
            return loss_dict
        else:
            # Normal inference mode (no targets)
            return self.base_model(images)


def train_one_epoch(model, optimizer, loader, device, epoch, writer, w_box, w_rpn, print_freq=50):
    model.train()
    for step, (images, targets) in enumerate(tqdm(loader, desc=f"Train E{epoch}")):
        t0 = time.time()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = (
            loss_dict['loss_classifier']
          + w_box * loss_dict['loss_box_reg']
          + loss_dict['loss_objectness']
          + w_rpn * loss_dict['loss_rpn_box_reg']
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        dt = time.time() - t0
        mem = torch.cuda.memory_allocated(device)/1024**2 if device.type=='cuda' else 0.0
        global_step = epoch * len(loader) + step
        writer.add_scalar('train/total_loss', total_loss.item(), global_step)
        for k, v in loss_dict.items():
            writer.add_scalar(f"train/{k}", v.item(), global_step)
        writer.add_scalar('train/iter_time', dt, global_step)
        writer.add_scalar('train/mem_MB', mem, global_step)

        if step % print_freq == 0:
            tqdm.write(f"Epoch {epoch} Step {step}/{len(loader)} - total_loss: {total_loss.item():.4f}")


def validate(model, loader, device, writer, epoch):
    model.eval()
    sum_dict = {k: 0.0 for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']}
    count = 0
    with torch.no_grad():
        for step, (images, targets) in enumerate(tqdm(loader, desc=f"Val E{epoch}")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            
            # With our wrapper, this will always return loss dictionaries
            loss_dict = model(images, targets)
            
            for k, v in loss_dict.items():
                sum_dict[k] += v.item()
            
            global_step = epoch * len(loader) + step
            for k, v in loss_dict.items():
                writer.add_scalar(f"val/{k}", v.item(), global_step)
            count += 1

    avg = {k: sum_dict[k]/count for k in sum_dict}
    print(f"[Epoch {epoch}] Val avg → cls: {avg['loss_classifier']:.4f}, box: {avg['loss_box_reg']:.4f}, obj: {avg['loss_objectness']:.4f}, rpn: {avg['loss_rpn_box_reg']:.4f}")
    return avg


def objective(trial):
    lr    = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    w_box = trial.suggest_float('w_box', 1.0, 5.0) # Multiplier for box regression loss, the larger the more focus on box regression
    w_rpn = trial.suggest_float('w_rpn', 1.0, 5.0) # Multiplier for RPN box regression loss, the larger the more focus on RPN box regression

    writer = SummaryWriter(log_dir=f"runs_RCNN/trial_{trial.number}")
    train_ds = YOLODataset(TRAIN_DIR)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_ds   = YOLODataset(VAL_DIR)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0, collate_fn=collate_fn)

    base_model = make_model(w_box, w_rpn).to(DEVICE)
    model = ModelWithCustomForward(base_model, w_box, w_rpn)  # Wrap the base model
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(1, 4):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, writer, w_box, w_rpn)
        lr_scheduler.step()
        avg_val = validate(model, val_loader, DEVICE, writer, epoch)
        trial.report(avg_val['loss_box_reg'], epoch)
        if trial.should_prune():
            writer.close()
            raise TrialPruned()

    writer.close()
    return avg_val['loss_box_reg']


def custom_train_eval(model, train_loader, val_loader, optimizer, lr_scheduler, writer, epochs, w_box, w_rpn, device):
    """Separated training function for cleaner execution after finding best params"""
    for epoch in range(1, epochs+1):
        # Train
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer, w_box, w_rpn)
        # Update learning rate
        lr_scheduler.step()
        # Validate
        validate(model, val_loader, device, writer, epoch)
        # Save checkpoint
        torch.save(model.base_model.state_dict(), f"fasterrcnn_final_epoch_{epoch}.pth")
        print(f"Saved model checkpoint for epoch {epoch}")


if __name__ == '__main__':
    try:
        # Run hyperparameter optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2)
        print('Best params:', study.best_trial.params)

        best = study.best_trial.params
        writer = SummaryWriter(log_dir='runs_RCNN/final')
        
        # Create datasets and loaders
        train_ds = YOLODataset(TRAIN_DIR)
        train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_ds = YOLODataset(VAL_DIR)
        val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        # Create model with best params
        base_model = make_model(best['w_box'], best['w_rpn']).to(DEVICE)
        model = ModelWithCustomForward(base_model, best['w_box'], best['w_rpn'])
        optimizer = torch.optim.Adam(base_model.parameters(), lr=best['lr'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Train with best parameters
        custom_train_eval(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            writer=writer,
            epochs=EPOCHS,
            w_box=best['w_box'],
            w_rpn=best['w_rpn'],
            device=DEVICE
        )
        
        writer.close()
        print('Training and validation complete.')
        
    except Exception as e:
        print(f"An error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        



    # ─── AFTER TRAINING ──────────────────────────────────────────────────────────

    # --- (1) Define a TEST_DIR, Dataset and DataLoader just like train/val ---
    TEST_DIR = r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\split_data\test"
    test_ds = YOLODataset(TEST_DIR)
    test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- (2) Load the best checkpoint you saved ---
    final_ckpt = f"fasterrcnn_final_epoch_{EPOCHS}.pth"
    print(f"Loading checkpoint from {final_ckpt}…")
    base_model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))

    # --- (3) Switch to eval mode and wrap for inference ---
    base_model.eval()
    infer_model = ModelWithCustomForward(base_model, w_box=best['w_box'], w_rpn=best['w_rpn']).to(DEVICE)

    # --- (4) Run a single batch (or loop over all) and print/save outputs ---
    import matplotlib.pyplot as plt
    from torchvision.utils import draw_bounding_boxes

    with torch.no_grad():
        for images, targets in test_loader:
            # --- inference on GPU ---
            images = [img.to(DEVICE) for img in images]
            outputs = infer_model(images)

            # --- prepare for plotting ---
            img = images[0]                 # still [C,H,W] on GPU
            img = (img * 255).to(torch.uint8)
            img = img.cpu()                 # <-- pull back to CPU
            boxes  = outputs[0]['boxes'].cpu()
            scores = outputs[0]['scores'].cpu()
            labels = outputs[0]['labels'].cpu()

            # filter by confidence threshold
            keep = scores > 0.5
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # draw and plot
            from torchvision.utils import draw_bounding_boxes
            import matplotlib.pyplot as plt

            vis = draw_bounding_boxes(img, boxes, 
                                    labels=[str(int(l)) for l in labels])
            plt.figure(figsize=(8,8))
            plt.imshow(vis.permute(1,2,0).numpy())  # now a CPU NumPy array
            plt.axis('off')
            plt.show()

            break


