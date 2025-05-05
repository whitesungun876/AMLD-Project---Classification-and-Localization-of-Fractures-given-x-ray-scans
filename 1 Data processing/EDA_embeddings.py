
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from scipy.stats import zscore

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.ensemble import IsolationForest


cwd = os.getcwd()
# Define paths
fractured_path = os.path.join(cwd, "Data","FracAtlas","images","Fractured")
non_fractured_path = os.path.join(cwd, "Data","FracAtlas","images","Non_fractured")

def load_images_from_folder(folder, size=(224, 224)):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(os.path.join(folder, filename)).convert('L')  # grayscale
                img = img.resize(size)
                img_array = np.array(img)
                images.append(img_array)
                filenames.append(filename)
            except Exception as e:
                print(f"Could not load {filename}: {e}")
    return np.array(images), filenames

fractured_imgs, fractured_filenames = load_images_from_folder(fractured_path)
nonfractured_imgs, nonfractured_filenames = load_images_from_folder(non_fractured_path)

def detect_pixel_outliers(images, threshold=3.0):
    flat_images = images.reshape(images.shape[0], -1)
    z_scores = np.abs(zscore(flat_images, axis=0))
    outlier_scores = z_scores.mean(axis=1)  # Mean z-score per image
    outliers = np.where(outlier_scores > threshold)[0]
    return outliers, outlier_scores

fractured_outliers, fractured_scores = detect_pixel_outliers(fractured_imgs)
nonfractured_outliers, nonfractured_scores = detect_pixel_outliers(nonfractured_imgs)

# Load pretrained model
model = resnet18(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last classification layer

# Image transform
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

def get_embeddings(image_folder):
    embeddings = []
    filenames = []
    for file in os.listdir(image_folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_folder, file)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                embedding = model(image).squeeze().numpy()
            embeddings.append(embedding)
            filenames.append(file)
    return np.array(embeddings), filenames

fractured_embeddings, fractured_names = get_embeddings(fractured_path)

# Detect outliers with Isolation Forest
iso = IsolationForest(contamination=0.05)
fractured_preds = iso.fit_predict(fractured_embeddings)
fractured_outliers = np.where(fractured_preds == -1)[0]

def show_outliers(images, filenames, indices):
    for i in indices:
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Outlier: {filenames[i]}')
        plt.axis('off')
        plt.show()

# For pixel-level
show_outliers(fractured_imgs, fractured_filenames, fractured_outliers[:5])