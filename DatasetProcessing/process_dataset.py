import cv2
import os
import glob
import torch
import random
import numpy as np

from torchvision.transforms import ToTensor

# Set torch float type
torch.set_default_dtype(torch.float32)

# Paths
img_dir = "face_images/*.jpg"  # Adjust if needed
augmented_dir = "augmented"
lab_dirs = {
    "L": "lab/L",
    "a": "lab/a",
    "b": "lab/b",
    "augmented" : "lab/augmented"
}

# Create folders if they do not exist
os.makedirs(augmented_dir, exist_ok=True)
for folder in lab_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Load original images and resize to 128x128
files = glob.glob(img_dir)
original_images = []

for f in files:
    img = cv2.imread(f)  # BGR read in
    img = cv2.resize(img, (128, 128))
    original_images.append(img)

original_images = np.array(original_images)
print(f"Loaded {len(original_images)} images")

# Convert to tensor: [n_images, C, H, W]
original_tensor = torch.stack([ToTensor()(img) for img in original_images])
original_tensor = original_tensor.permute(0, 1, 2, 3)  # Still [N, C, H, W]

# Shuffle images
perm = torch.randperm(original_tensor.size(0))
original_tensor = original_tensor[perm]

# Data augmentation factor
AUG_FACTOR = 10
augmented_images = []

def augment_image(img):
    # Random flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Random crop and resize
    h, w = img.shape[:2]
    crop_size = random.randint(int(0.8 * h), h)
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    cropped = img[y:y+crop_size, x:x+crop_size]
    cropped = cv2.resize(cropped, (128, 128))

    # Random brightness scale
    scale = random.uniform(0.6, 1.0)
    scaled = np.clip(cropped * scale, 0, 255).astype(np.uint8)

    return scaled

# Apply augmentation
counter = 0
for i in range(len(original_images)):
    for j in range(AUG_FACTOR):
        aug_img = augment_image(original_images[i])
        augmented_images.append(aug_img)

        # Save RGB version
        filename = f"aug_{i}_{j}.jpg"
        cv2.imwrite(os.path.join(augmented_dir, filename), aug_img)

        # Convert to LAB
        lab_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab_img)

        # Save each channel
        cv2.imwrite(os.path.join(lab_dirs["L"], f"L_{i}_{j}.jpg"), L)

        # Convert single channel a* and b* to 3-channel RGB (for visualization)
        a_corrected = cv2.merge([np.full_like(L, 128), a, np.full_like(b, 128)])
        a_rgb = cv2.cvtColor(a_corrected, cv2.COLOR_LAB2RGB)
        b_corrected = cv2.merge([np.full_like(L, 128), np.full_like(a, 128), b])
        b_rgb = cv2.cvtColor(b_corrected, cv2.COLOR_LAB2RGB)
        cv2.imwrite(os.path.join(lab_dirs["a"], f"a_{i}_{j}.jpg"), cv2.cvtColor(a_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(lab_dirs["b"], f"b_{i}_{j}.jpg"), cv2.cvtColor(b_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(lab_dirs["augmented"], f"lab_aug_{i}_{j}.jpg"), lab_img)

        counter += 1

print(f"Generated and saved {counter} augmented images with LAB splits.")
