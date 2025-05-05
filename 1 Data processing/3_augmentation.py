
'''
Author: Søren

Date: 22/04/2025

Description: Data augmentation

'''

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os

# Load Image
image_path = "/path/to/your/image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cwd = os.getcwd() #cwd should be amld2025_fracAtlas
running_script = False # True if this script is run as a whole, False if it is run in parts (line by line)
print("cwd: ", cwd)




"C:\Users\Soren\Documents\amld2025_fracAtlas\Data\FracAtlas\Annotations\YOLO"

# Prepare Bounding Boxes (example using 'coco' format)
# Each inner list is [x_min, y_min, bbox_width, bbox_height]
bboxes = np.array([
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
])

# Prepare Labels (using the name specified in label_fields)
class_labels = ['dog', 'cat', 'sports ball']
# Example with multiple label fields if defined in BboxParams:
# class_categories = ['animal', 'animal', 'item']




transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Blur(p=0.1),
    A.Rotate(limit=15, p=0.5),
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
)

# Apply
transformed = transform(image=image, bboxes=bboxes, category_ids=labels)