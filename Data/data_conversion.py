import os
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Constants
CWD = os.getcwd()  # Change this if needed
TEST_DIR = os.path.join(CWD, "Data", "split_data", "test", "images")
OUTPUT_BASE = os.path.join(CWD, "Data", "split_data")

# Define transformations
transformations = {
    "brightness_minus_50": lambda img: ImageEnhance.Brightness(img).enhance(0.5),
    "brightness_plus_50": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
    "contrast_minus_50": lambda img: ImageEnhance.Contrast(img).enhance(0.5),
    "contrast_plus_50": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
}

# Apply transformations
for name, transform in tqdm(transformations.items()):
    out_dir = os.path.join(OUTPUT_BASE, "test_"+name, "images")
    os.makedirs(out_dir,  exist_ok=True)

    for fname in os.listdir(TEST_DIR):
        if fname.lower().endswith(".jpg"):
            path = os.path.join(TEST_DIR, fname)
            image = Image.open(path)
            transformed = transform(image)
            transformed.save(os.path.join(out_dir, fname))

print("Transformation complete.")