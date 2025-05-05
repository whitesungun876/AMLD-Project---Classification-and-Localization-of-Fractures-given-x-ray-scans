import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
TEST_DIR    = Path(r"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\split_data\test")
WEIGHTS     = Path("fasterrcnn_final_epoch_2.pth")  # path to your saved weights
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESH = 0.5
BATCH       = 8

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
        target = {"boxes": boxes, "labels": labels}
        return image, target

# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
def make_model():
    """Recreate Faster R-CNN with the same RPN anchor generator as during training."""
    # Anchor generator matching training: 5 FPN levels, sizes (8,16,32) and ratios (0.5,1.0,2.0)
    anchor_gen = AnchorGenerator(
        sizes=((8, 16, 32),) * 5,
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_gen
    )
    # Replace ROI head predictor for 2 classes
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes=2)
    return model

# ─── EVALUATION UTILITIES ────────────────────────────────────────────────────
def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """Compute COCO mAP@IoU=0.5 and class precision."""
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], class_metrics=True)
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            inputs = [img.to(device) for img, _ in zip(images, targets)]
            outputs = model(inputs)
            preds, gts = [], []
            for out, (_, tgt) in zip(outputs, zip(images, targets)):
                preds.append({
                    'boxes': out['boxes'].cpu(),
                    'scores': out['scores'].cpu(),
                    'labels': out['labels'].cpu()
                })
                gts.append({
                    'boxes': tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu()
                })
            metric.update(preds, gts)
    return metric.compute()


def visualize_detections(model, image_paths, device, conf_thresh=0.5):
    """Visualize model detections on a list of image file paths."""
    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            img = read_image(str(img_path)).to(torch.float32) / 255.0
            out = model([img.to(device)])[0]
            boxes = out['boxes'].cpu()
            scores = out['scores'].cpu()
            labels = out['labels'].cpu()
            keep = scores > conf_thresh
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            vis = draw_bounding_boxes(
                (img * 255).to(torch.uint8),
                boxes,
                labels=[f"{int(l)}:{s:.2f}" for l, s in zip(labels, scores)]
            )
            plt.figure(figsize=(6,6))
            plt.title(img_path.name)
            plt.imshow(vis.permute(1,2,0).numpy())
            plt.axis('off')
            plt.show()

# ─── MAIN: TEST EVALUATION ONLY ───────────────────────────────────────────────
if __name__ == '__main__':
    # Load model and weights
    model = make_model().to(DEVICE)
    checkpoint = torch.load(WEIGHTS, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    # Prepare test loader
    test_ds = YOLODataset(TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Quantitative evaluation
    metrics = evaluate_model(model, test_loader, DEVICE, iou_threshold=0.5)
    print(f"mAP@0.5: {metrics['map_50']:.4f}")
    if 'class_precision' in metrics and metrics['class_precision'].numel() > 0:
        prec = metrics['class_precision'][0].item()
        print(f"Classification error (1 - class precision): {1 - prec:.4f}")
    else:
        print("Warning: 'class_precision' not available; classification error not computed.")

    # Visualize first 10 test images
    img_paths = [Path(TEST_DIR)/'images'/p.name for p in test_ds.image_files[:10]]
    visualize_detections(model, img_paths, DEVICE, conf_thresh=CONF_THRESH)
