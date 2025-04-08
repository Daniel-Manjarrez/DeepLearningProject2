import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision.transforms import ToTensor
import random

import glob

# Set default tensor type
torch.set_default_dtype(torch.float32)

# ==== COLORIZATION CNN ====
class ColorizationCNN(nn.Module):
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()

        # Downsampling with Convolution, Batch Normalization
        self.down1 = self.upconv_layer(1, 64)
        self.down2 = self.upconv_layer(64, 128)
        self.down3 = self.upconv_layer(128, 256)
        self.down4 = self.upconv_layer(256, 512)
        self.down5 = self.upconv_layer(512, 512)
        
        self.up5 = self.downconv_layer(512, 512)
        self.up4 = self.downconv_layer(512, 256)
        self.up3 = self.downconv_layer(256, 128)
        self.up2 = self.downconv_layer(128, 64)
        self.up1 = self.downconv_layer(64, 2)  # Output 2 channels for a* and b*
        
        ''' Part 5 hyperparameter tuning attempts 
        
        # Increasing the number of channels in the downsampling layers
        # and decreasing in the upsampling layers
        # self.down1 = self.upconv_layer(1, 32)   # Reduced from 64 to 32
        # self.down2 = self.upconv_layer(32, 64)  # Reduced from 128 to 64
        # self.down3 = self.upconv_layer(64, 128) # Reduced from 256 to 128
        # self.down4 = self.upconv_layer(128, 256) # Reduced from 512 to 256
        # self.down5 = self.upconv_layer(256, 256) # Reduced from 512 to 256

        # self.up5 = self.downconv_layer(256, 256) # Reduced from 512 to 256
        # self.up4 = self.downconv_layer(256, 128) # Reduced from 256 to 128
        # self.up3 = self.downconv_layer(128, 64)  # Reduced from 128 to 64
        # self.up2 = self.downconv_layer(64, 32)   # Reduced from 64 to 32
        # self.up1 = self.downconv_layer(32, 2)    # Output remains 2 channels for a* and b*
        
        # Decrease the number of channels in the downsampling layers
        # and increase in the upsampling layers
        # self.down1 = self.upconv_layer(1, 16)   # Reduced from 64 to 16
        # self.down2 = self.upconv_layer(16, 32)  # Reduced from 128 to 32
        # self.down3 = self.upconv_layer(32, 64)  # Reduced from 256 to 64
        # self.down4 = self.upconv_layer(64, 128) # Reduced from 512 to 128
        # self.down5 = self.upconv_layer(128, 256) # Reduced from 512 to 256
        
        # self.up5 = self.downconv_layer(256, 128) # Reduced from 512 to 128
        # self.up4 = self.downconv_layer(128, 64)  # Reduced from 256 to 64
        # self.up3 = self.downconv_layer(64, 32)   # Reduced from 128 to 32
        # self.up2 = self.downconv_layer(32, 16)   # Reduced from 64 to 16
        # self.up1 = self.downconv_layer(16, 2)  # Output remains 2 channels for a* and b*
        
        '''
    
    def upconv_layer(self, in_channels, out_channels, ksize = 3, stride = 2, padding =1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def downconv_layer(self, in_channels, out_channels, ksize=4, stride=2, padding=1, is_tanh=False):
        if not is_tanh:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.Tanh()
            )
    
    def forward(self, x):
        # Performing downsampling
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        # Performing upsampling
        x = self.up5(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        
        return x

# ==== DATASET ====
class ColorizationDataset(Dataset):
    def __init__(self, gray_image_paths, color_image_paths, augment=False):
        self.gray_image_paths = gray_image_paths
        self.color_image_paths = color_image_paths
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        ])

    def __len__(self):
        return len(self.gray_image_paths)

    def __getitem__(self, idx):
        # Load grayscale image
        gray_img = Image.open(self.gray_image_paths[idx]).convert("L")  # Grayscale
        gray_img = self.transform(gray_img)

        # Load color image
        color_img = Image.open(self.color_image_paths[idx]).convert("RGB")  # Color
        color_img = self.transform(color_img)

        # Convert color image to LAB
        lab = rgb2lab(np.array(color_img)).astype("float32")
        lab[:, :, 0] /= 100.0      # Normalize L to [0, 1]
        lab[:, :, 1:] /= 128.0    # Normalize ab to [-1, 1]

        L = np.array(gray_img) / 255.0  # Normalize grayscale image to [0, 1]
        L = np.expand_dims(L, axis=-1)  # Add channel dimension
        ab = lab[:, :, 1:]

        L_tensor = torch.from_numpy(L).permute(2, 0, 1).float()  # Shape: [1, H, W]
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1).float()  # Shape: [2, H, W]

        return L_tensor, ab_tensor

# ==== TEST, EVALUATE, AND SAVE MODEL IMAGES ====
def evaluate_and_save(model, test_loader, device, output_folder="PredictedColorizedImg"):
    # make the PredictedColorizedImg directory if it exist
    os.makedirs(output_folder, exist_ok=True)

    # Evaluate the model 
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, (L_batch, ab_batch) in enumerate(test_loader):
            L_batch = L_batch.to(device)
            ab_batch = ab_batch.to(device)
            preds = model(L_batch)
            loss = nn.functional.mse_loss(preds, ab_batch, reduction='mean')
            total_loss += loss.item()
            # count += L_batch.size(0)
            count += 1

            # Save RGB colorized results
            for j in range(L_batch.size(0)):
                L = L_batch[j].cpu().numpy().squeeze() * 100
                ab = preds[j].cpu().numpy().transpose(1, 2, 0) * 128
                lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
                lab[:, :, 0] = L
                lab[:, :, 1:] = ab
                rgb = lab2rgb(lab)
                rgb_img = (rgb * 255).astype(np.uint8)
                img = Image.fromarray(rgb_img)
                img.save(os.path.join(output_folder, f"test_img_{i*10 + j}.png"))

    # mse = total_loss / (count * 2 * 640 * 480)

    # Print out mse loss of the testing
    mse = total_loss / count
    print(f"Test MSE: {mse:.6f}")

# ==== GENERATE AUGMENTED IMAGES ====
def generate_augmented_images(img_to_aug):
        # Set torch float type
    torch.set_default_dtype(torch.float32)

    # Paths
    augmented_dir = "augmented"

    # Create folders if they do not exist
    os.makedirs(augmented_dir, exist_ok=True)

    # Read in every image and resize them to 128 by 128
    original_images = []
    for f in img_to_aug:
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

            counter += 1

    print(f"Generated and saved {counter} augmented images with LAB splits.")

# ==== LOAD PRETRAINED MODEL FOR NCDataset ====
def load_pretrained_model(model, model_path, device):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Loaded pretrained model weights from {model_path}")
    else:
        print(f"Model path {model_path} does not exist. Starting from scratch.")
    return model

# === GET NCDataset IMAGE PATHS ====
def get_image_paths_from_directory(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def transfer_learning_on_ncdataset(model, device, gray_dir, color_dir, num_epochs=10, batch_size=10, learning_rate=1e-4):
    # Fetch paired grayscale and color image paths
    gray_image_paths, color_image_paths = get_paired_image_paths(gray_dir, color_dir)

    # Split into training (90%) and testing (10%) datasets
    train_size = int(0.9 * len(gray_image_paths))
    test_size = len(gray_image_paths) - train_size
    train_gray, test_gray = random_split(gray_image_paths, [train_size, test_size])
    train_color, test_color = random_split(color_image_paths, [train_size, test_size])

    # Create datasets and dataloaders
    train_dataset = ColorizationDataset(train_gray, train_color)
    test_dataset = ColorizationDataset(test_gray, test_color)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Fine-tune the model
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            preds = model(L_batch)
            loss = criterion(preds, ab_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}")

    # Evaluate the model on the test set
    evaluate_and_save(model, test_loader, device=device, output_folder="PredictedColorizedImg_NCDataset")

def freeze_layers(model, num_layers_to_freeze):
    """
    Freeze the first `num_layers_to_freeze` layers of the model.
    """
    layers = list(model.children())  # Get all layers of the model
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

def get_paired_image_paths(gray_dir, color_dir):
    gray_image_paths = []
    color_image_paths = []

    for root, _, files in os.walk(gray_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                gray_path = os.path.join(root, file)
                color_path = os.path.join(color_dir, os.path.relpath(gray_path, gray_dir))
                if os.path.exists(color_path):
                    gray_image_paths.append(gray_path)
                    color_image_paths.append(color_path)

    return gray_image_paths, color_image_paths

# ==== MAIN ====
def main():
    base_dir = os.getcwd()
    gray_dir = os.path.join(base_dir, "Gray")
    color_dir = os.path.join(base_dir, "ColorfulOriginal")

    # Load the pretrained model
    model_path = os.path.join(base_dir, "colorization_model_face.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationCNN()
    model = load_pretrained_model(model, model_path, device)

    # Freeze the first 4 layers of the model
    freeze_layers(model, num_layers_to_freeze=4)

    # Perform transfer learning on the NCDataset
    transfer_learning_on_ncdataset(model, device, gray_dir, color_dir)

if __name__ == "__main__":
    main()
