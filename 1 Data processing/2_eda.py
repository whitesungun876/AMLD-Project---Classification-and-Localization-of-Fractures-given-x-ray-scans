'''
Author: Sebastian Faurby

Descrition: Explorative data analysis
'''

import numpy as np
import pandas as pd
import random
import torch
import os
import sys
from PIL import Image
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from sklearn.model_selection import train_test_split
import seaborn as sns
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import random
from torchvision.models import densenet121
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA



cwd = os.getcwd() #cwd should be amld2025_fracAtlas
# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(cwd, '..')))
import functions


###############################################################################
##### Central configuration
###############################################################################

# Define paths
fracAtlas = cwd + "\\Data\\fracAtlas\\" #path to fracAtlas folder
all_frac_folder = fracAtlas + "\\images\\all\\"
annotation_path = fracAtlas + "\\Annotations\\YOLO\\" #path to yolo annotations folder
fractured_path = fracAtlas + "\\images\\Fractured\\"
non_fractured_path = fracAtlas + "\\images\\Non_fractured\\"
preproc_folder = cwd + "\\Data\\data_rgb\\"
graphics_folder = cwd + "\\graphics\\eda\\"
# Create output directories
output_dir = cwd + "\\Data\\split_data"
os.makedirs(output_dir, exist_ok=True)

###############################################################################
##### Define functions
###############################################################################

def distribution_bar(df, col, ticklabels = ['Non-fractured', 'Fractured'],
                    title = 'Distribution of labels',
                    save_path= None):
    """
    Plot the distribution of a column in a dataframe as a bar plot.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df[col].value_counts()
    bars = counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Count')

    # Rotate x labels
    ax.set_xticklabels(ticklabels, rotation=0)

    # Add thousands separator to bar labels
    for bar in bars.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:,}',  # Add thousands separator
            ha='center',
            va='bottom'
        )

    fig.tight_layout()
    plt.show()

def distribution_bar_pct(df, col, ticklabels = ['Non-fractured', 'Fractured'],
                    title = 'Distribution of labels',
                    save_path= None):
    """
    Plot the distribution of a column in a dataframe as a bar plot.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    # Calculate the total number of observations
    total_counts = df[col].value_counts().sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df[col].value_counts()
    bars = counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Count')

    # Rotate x labels
    ax.set_xticklabels(['Non-fractured', 'Fractured'], rotation=0)

    # Add percentage labels to the bars
    for bar in bars.patches:
        value = bar.get_height()
        percentage = (value / total_counts) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{percentage:.1f}%',  # Format as percentage with one decimal place
            ha='center',
            va='bottom'
        )

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path+".pdf")
    else:
        plt.show()

###############################################################################
##### Load the data
###############################################################################


df_meta = pd.read_csv(fracAtlas + 'dataset.csv')
df_meta = df_meta.rename(columns={'fractured': 'label'}) #rename the column to label

data_split = functions.load_data(cwd = cwd,
                            return_torch = False,
                            batch_size = 32,
                            rgb = True)

df_train = data_split['train_split']
df_val = data_split['val_split']
df_test = data_split['test_split']

df_all = [df_train, df_val, df_test]

# for all dfs drop last column
for df in df_all:
    df.drop(columns = ['img_path'], inplace = True)
    df.reset_index(drop=True, inplace = True)


###############################################################################
##### Descriptive statistics of the dataset
###############################################################################

## plot the distribution of the labels in the dataset

# Calculate the total number of observations
total_counts = df_meta['label'].value_counts().sum()

fig, ax = plt.subplots(figsize=(8, 6))
counts = df_meta['label'].value_counts()
bars = counts.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')

# Rotate x labels
ax.set_xticklabels(['Non-fractured', 'Fractured'], rotation=0)

# Add percentage labels to the bars
for bar in bars.patches:
    value = bar.get_height()
    percentage = (value / total_counts) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as percentage with one decimal place
        ha='center',
        va='bottom'
    )

fig.tight_layout()
plt.show()

# Plotting the distribution based on the fracture type, hand, leg, hip, shoulder or mixed
# Making the binary variables hand, leg, hip, should and mixed into one column called fracture_type
# and plotting the distribution of the fracture type

df_non_frac = df_meta[df_meta['label'] == 0] #keep only the non-fractured images

df_non_frac['body_part'] = df_non_frac[['hand', 'leg', 'hip', 'shoulder']].idxmax(axis=1)
df_non_frac['body_part'] = df_non_frac['body_part'].replace({'hand': 'Hand', 'leg': 'Leg', 'hip': 'Hip', 'shoulder': 'Shoulder'})

fig, ax = plt.subplots(figsize=(8, 6))
counts_type = df_non_frac['body_part'].value_counts()
bars = counts_type.plot(kind='bar', ax=ax)
ax.set_ylabel('Count')

# Rotate x labels, Creating title and labels
ax.set_xticklabels(['Hand', 'Leg', 'Hip', 'Shoulder'], rotation=0)
ax.set_title('Distribution in Non-Fractured Images')
# Removing the x-axis label
ax.set_xlabel('')

# Add percentage labels to the bars
for bar in bars.patches:
    value = bar.get_height()
    percentage = (value / total_counts) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{percentage:.1f}%',  # Format as percentage with one decimal place
        ha='center',
        va='bottom'
    )


fig.tight_layout()
plt.show()


# Keep only the fractured images
df_frac = df_meta[df_meta['label'] == 1]

# Determine the dominant body part for each image
df_frac['body_part'] = df_frac[['hand', 'leg', 'hip', 'shoulder']].idxmax(axis=1)
df_frac['body_part'] = df_frac['body_part'].replace({
    'hand': 'Hand',
    'leg': 'Leg',
    'hip': 'Hip',
    'shoulder': 'Shoulder'
})

# Compute counts and total for percentages
counts_type = df_frac['body_part'].value_counts()
total_counts = counts_type.sum()

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = counts_type.plot(kind='bar', ax=ax)

# Labels and title
ax.set_ylabel('Count')
ax.set_xlabel('')
ax.set_title('Distribution in Fractured Images')
ax.set_xticklabels(['Hand', 'Leg', 'Hip', 'Shoulder'], rotation=0)

# Add percentage labels on top of bars
for bar in bars.patches:
    height = bar.get_height()
    pct = (height / total_counts) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{pct:.1f}%',
        ha='center',
        va='bottom'
    )

fig.tight_layout()
plt.show()



# Counting the number of images with bounding boxes in the fractured images

def plot_bbox_count_distribution(frac_dir, annotation_dir):
    counts = []
    for filename in os.listdir(frac_dir):
        if filename.endswith('.jpg'):
            label_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path) as f:
                    lines = f.readlines()
                    counts.append(len(lines))

    plt.hist(counts, bins=range(1, max(counts)+2), edgecolor='black', color='skyblue', align='left')
    plt.title("Distribution of Bounding Boxes per Fractured Image")
    plt.xlabel("Bounding Boxes")
    plt.ylabel("Number of Images")
    plt.show()

plot_bbox_count_distribution(
    os.path.join(fracAtlas, 'images', 'Fractured'),
    annotation_path
)

# Printing examples of a 5 fractured and 5 non-fractured images

def load_yolo_boxes(label_path, img_size):
    w_img, h_img = img_size
    boxes = []

    if not os.path.exists(label_path):
        return boxes  # No label file, return empty

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id, x_center, y_center, width, height = map(float, parts)
                x_center *= w_img
                y_center *= h_img
                width *= w_img
                height *= h_img
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                boxes.append((x_min, y_min, width, height, int(cls_id)))

    return boxes

def plot_selected_images_with_bboxes(frac_list, nonfrac_list, frac_dir, nonfrac_dir, annotation_dir):
    all_imgs = [(img, 'Fractured') for img in frac_list] + [(img, 'Non Fractured') for img in nonfrac_list]

    plt.figure(figsize=(20, 8))
    
    for i, (img_name, label) in enumerate(all_imgs):
        folder = frac_dir if label == 'Fractured' else nonfrac_dir
        img_path = os.path.join(folder, img_name)
        lbl_path = os.path.join(annotation_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size
        boxes = load_yolo_boxes(lbl_path, (w_img, h_img))

        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(img)
        for (x, y, w, h, cls_id) in boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x - 10, y - 80, f"Class {cls_id}", color='red', fontsize=8, backgroundcolor='white')

        ax.set_title(f"{label}\n{img_name}", fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    

fractured_images = [
    'IMG0000019.jpg',
    'IMG0000025.jpg',
    'IMG0002330.jpg',
    'IMG0000058.jpg',
    'IMG0002312.jpg'
]

nonfractured_images = [
    'IMG0000001.jpg',
    'IMG0000011.jpg',
    'IMG0000026.jpg',
    'IMG0000048.jpg',
    'IMG0000130.jpg'
]

plot_selected_images_with_bboxes(
    frac_list=fractured_images,
    nonfrac_list=nonfractured_images,
    frac_dir=os.path.join(fracAtlas, 'images', 'Fractured'),
    nonfrac_dir=os.path.join(fracAtlas, 'images', 'Non_fractured'),
    annotation_dir=annotation_path
)

# Bouding box size distribution

def plot_bbox_size_distribution(frac_dir, annotation_dir):
    areas = []
    for filename in os.listdir(frac_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(frac_dir, filename)
            label_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            
            img = Image.open(img_path)
            w_img, h_img = img.size
            boxes = load_yolo_boxes(label_path, (w_img, h_img))

            for (_, _, w, h, _) in boxes:
                areas.append(w * h)

    plt.hist(areas, bins=30, color='orange', edgecolor='black')
    plt.title("Distribution of Bounding Box Areas")
    plt.xlabel("Box Area (pixels²)")
    plt.ylabel("Frequency")
    plt.show()
    
    
plot_bbox_size_distribution(
    os.path.join(fracAtlas, 'images', 'Fractured'),
    annotation_path
)


# Images with large boxes

def get_images_with_large_bboxes(frac_dir, annotation_dir, area_threshold=10000):
    images_with_large_boxes = []

    for filename in os.listdir(frac_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(frac_dir, filename)
            label_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            
            img = Image.open(img_path)
            w_img, h_img = img.size
            boxes = load_yolo_boxes(label_path, (w_img, h_img))

            for (_, _, w, h, _) in boxes:
                area = w * h
                if area > area_threshold:
                    images_with_large_boxes.append(filename)
                    break  # Only add once even if multiple large boxes
    return images_with_large_boxes

large_box_images = get_images_with_large_bboxes(
    frac_dir=os.path.join(fracAtlas, 'images', 'Fractured'),
    annotation_dir=annotation_path,
    area_threshold=10000
)

print(f"{len(large_box_images)} images found with boxes > 10,000 px²:")
for name in large_box_images:
    print(name)
    
# Sorting the large boxes by size   
    
def get_large_bboxes_sorted(frac_dir, annotation_dir, area_threshold=10000):
    large_boxes = []

    for filename in os.listdir(frac_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(frac_dir, filename)
            label_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            
            img = Image.open(img_path)
            w_img, h_img = img.size
            boxes = load_yolo_boxes(label_path, (w_img, h_img))

            for (_, _, w, h, _) in boxes:
                area = w * h
                if area > area_threshold:
                    large_boxes.append((filename, int(area)))

    # Sort descending by area
    large_boxes.sort(key=lambda x: x[1], reverse=True)
    return large_boxes    


large_boxes_sorted = get_large_bboxes_sorted(
    frac_dir=os.path.join(fracAtlas, 'images', 'Fractured'),
    annotation_dir=annotation_path,
    area_threshold=10000
)

print(f"Found {len(large_boxes_sorted)} bounding boxes with area > 10,000 px²:\n")
for filename, area in large_boxes_sorted:
    print(f"{filename} — {area} px²")
    
    

## Checking the dimension of the images

def check_image_dimensions(image_dir):
    dimensions = {}
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            size = img.size
            dimensions[size] = dimensions.get(size, 0) + 1

    for dim, count in dimensions.items():
        print(f"Size {dim[0]}x{dim[1]}: {count} images")
        


print("Fractured:")
check_image_dimensions(os.path.join(fracAtlas, 'images', 'Fractured'))

print("\nNon Fractured:")
check_image_dimensions(os.path.join(fracAtlas, 'images', 'Non_fractured'))


## Plotting the pixel intensitiets

# The reason this make sense is that 
# A low pixel intensity means that we have soft tissue (black) and a high pixel intensity means that we have bone (white).
# The fractured images should have a higher pixel intensity than the non-fractured images, since the fractures are in the bone.

# This is also due to implatns, or severe breaks in the bone, which will be visible as white spots in the image.
# The pixel intensity is a measure of the brightness of the pixel, where 0 is black and 255 is white.

def analyze_grayscale_intensity(frac_list, nonfrac_list, frac_dir, nonfrac_dir, bins=50):
    import numpy as np

    frac_pixels = []
    nonfrac_pixels = []

    # Fractured images
    for img_name in frac_list:
        img_path = os.path.join(frac_dir, img_name)
        img = Image.open(img_path).convert('L')  # grayscale
        frac_pixels.extend(np.array(img).flatten())

    # Non-fractured images
    for img_name in nonfrac_list:
        img_path = os.path.join(nonfrac_dir, img_name)
        img = Image.open(img_path).convert('L')  # grayscale
        nonfrac_pixels.extend(np.array(img).flatten())

    # Convert to NumPy arrays
    frac_pixels = np.array(frac_pixels)
    nonfrac_pixels = np.array(nonfrac_pixels)

    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(frac_pixels, bins=bins, alpha=0.6, label='Fractured', color='red', density=True)
    plt.hist(nonfrac_pixels, bins=bins, alpha=0.6, label='Not Fractured', color='green', density=True)
    plt.title("Grayscale Pixel Intensity Distributions")
    plt.xlabel("Pixel Intensity (0=black, 255=white)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute and print means
    # It would be expected to have a higher mean of pixels intensity in the fractured images than in the non-fractured images.
    mean_frac = frac_pixels.mean()
    mean_nonfrac = nonfrac_pixels.mean()

    print(f"📊 Mean pixel intensity (Fractured):     {mean_frac:.2f}")
    print(f"📊 Mean pixel intensity (Not Fractured): {mean_nonfrac:.2f}")

fractured_images = [
    'IMG0000019.jpg',
    'IMG0000025.jpg',
    'IMG0002330.jpg',
    'IMG0000058.jpg',
    'IMG0002312.jpg',
    'IMG0001981.jpg',
    'IMG0002369.jpg',
    'IMG0002497.jpg',
    'IMG0002512.jpg',
    'IMG0002522.jpg',
    'IMG0001763.jpg',
    'IMG0001934.jpg',
    'IMG0002379.jpg',
    'IMG0002441.jpg',
    'IMG0002505.jpg',
    'IMG0002549.jpg',
    'IMG0002574.jpg',
    'IMG0003146.jpg',
    'IMG0003509.jpg',
    'IMG0004266.jpg'
]

nonfractured_images = [
    'IMG0000001.jpg',
    'IMG0000011.jpg',
    'IMG0000026.jpg',
    'IMG0000048.jpg',
    'IMG0000130.jpg',
    'IMG0003031.jpg',
    'IMG0003067.jpg',
    'IMG0003097.jpg',
    'IMG0003403.jpg',
    'IMG0003991.jpg',
    'IMG0000211.jpg',
    'IMG0000287.jpg',
    'IMG0000399.jpg',
    'IMG0000552.jpg',
    'IMG0000632.jpg',
    'IMG0000757.jpg',
    'IMG0000883.jpg',
    'IMG0001173.jpg',
    'IMG0001313.jpg',
    'IMG0001912.jpg'
]

    
analyze_grayscale_intensity(
    frac_list=fractured_images,
    nonfrac_list=nonfractured_images,
    frac_dir=os.path.join(fracAtlas, 'images', 'Fractured'),
    nonfrac_dir=os.path.join(fracAtlas, 'images', 'Non_Fractured'),
    bins=50
)


### t-SNE visualization of the images


# Load all images from the fractured and non-fractured directories
fractured_full_list = os.listdir(fractured_path)
nonfractured_full_list = os.listdir(non_fractured_path)

# Sample 100 .jpg images from each class
#fractured_images = random.sample([f for f in fractured_full_list if f.endswith('.jpg')], 100)
#nonfractured_images = random.sample([f for f in nonfractured_full_list if f.endswith('.jpg')], 100)

# Use pretrained densenet121 model for feature extraction from RadImageNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = densenet121(pretrained=False)

# 2) load your RadImageNet‑trained weights (you’d need to convert the .h5 → .pth)
state_dict = torch.load(cwd + "/Pre-trained weights/DenseNet121.pt", map_location=device)
_= model.load_state_dict(state_dict)

# 3) strip off the classifier, and pool to a 1×1 feature map
feature_extractor = torch.nn.Sequential(
    model.features,
    torch.nn.ReLU(inplace=True),
    torch.nn.AdaptiveAvgPool2d((1,1)),
    torch.nn.Flatten()
).to(device).eval()


# Preprocessing: resize to 416x416 and normalize
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])



def extract_embeddings(image_list, folder, label):
    embeddings = []
    labels = []
    filenames = []
    for img_name in image_list:
        path = os.path.join(folder, img_name)
        img = Image.open(path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(input_tensor).squeeze().cpu().numpy()
        embeddings.append(features)
        labels.append(label)
        filenames.append(img_name)
    return embeddings, labels, filenames




def plot_tsne(embeddings, labels):
    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2,
                perplexity=min(30, len(embeddings) // 3),
                random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    # compute indices for each class just once
    frac_idx    = [i for i, l in enumerate(labels) if l == "Fractured"]
    nonfrac_idx = [i for i, l in enumerate(labels) if l == "Non Fractured"]

    # plot each class exactly one time
    plt.scatter(reduced[frac_idx, 0], reduced[frac_idx, 1],
                c='orange', label='Fractured',   alpha=0.6, zorder=1)
    plt.scatter(reduced[nonfrac_idx, 0], reduced[nonfrac_idx, 1],
                c='blue',   label='Non Fractured', alpha=0.6, zorder=2)

    plt.legend()
    plt.title("t-SNE of Radiograph Embeddings")
    plt.show()

frac_embed, frac_labels, frac_files = extract_embeddings(fractured_full_list, fractured_path, "Fractured")
nonfrac_embed, nonfrac_labels, nonfrac_files = extract_embeddings(nonfractured_full_list, non_fractured_path, "Not Fractured")

all_embeddings = frac_embed + nonfrac_embed
all_labels = frac_labels + nonfrac_labels
all_files = frac_files + nonfrac_files

plot_tsne(all_embeddings, all_labels)



print(len(fractured_images), "fracture imgs sampled")
print(len(nonfractured_images), "non‑fracture imgs sampled")

# and even:
print("All files in non‑fract folder:", os.listdir(non_fractured_path))
print("Filtered .jpg only:", [f for f in os.listdir(non_fractured_path) if f.lower().endswith('.jpg')])


## Output - Similar embeddings will be placed close to eachother in a 2D space



### Next:

# PCA instead of t-SNE

def plot_pca(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)
    plt.title("PCA of Radiograph Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_pca(all_embeddings, all_labels)

# Detect outliers by distance from centroid
# Over / under exposed images

def detect_outliers_by_distance(embeddings, labels, filenames, threshold=1.5):
    embeddings = np.array(embeddings)
    outliers = []

    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        cluster = embeddings[idxs]
        centroid = cluster.mean(axis=0)
        dists = np.linalg.norm(cluster - centroid, axis=1)
        z_scores = (dists - dists.mean()) / dists.std()

        for i_idx, z in zip(idxs, z_scores):
            if z > threshold:
                outliers.append((filenames[i_idx], labels[i_idx], z))
    
    return outliers

outliers = detect_outliers_by_distance(all_embeddings, all_labels, all_files, threshold=1.5)

print("🔍 Outliers detected:")
for fname, label, z in outliers:
    print(f"{fname} | Label: {label} | Z-score: {z:.2f}")



# Autoencoder for image reconstruction 

# Self-supervised models (e.g., DINO, SimCLR) for feature extraction

# Show embeddings as heatmaps

# View embeddings as heatmaps using matplotlib.imshow (if reshaped)



# We need the distribution from the meta data (how can this be used to put more light on the fractured images?)