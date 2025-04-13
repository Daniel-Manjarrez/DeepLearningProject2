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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import glob
import warnings

warnings.filterwarnings("ignore", message=".*Conversion from CIE-LAB, via XYZ to sRGB color space resulted in.*")

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
class ColorfulOriginalDataset(Dataset):
    def __init__(self, color_image_paths, augment=False, noise_std=0.05):
        self.color_image_paths = color_image_paths
        self.augment = augment
        self.noise_std = noise_std  # Standard deviation of noise
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        ])

    def __len__(self):
        return len(self.color_image_paths)

    def __getitem__(self, idx):
        # Load color image
        color_img = Image.open(self.color_image_paths[idx]).convert("RGB")
        color_img = self.transform(color_img)

        # Convert color image to LAB
        lab = rgb2lab(np.array(color_img)).astype("float32")
        lab[:, :, 0] /= 100.0      # Normalize L to [0, 1]
        lab[:, :, 1:] /= 128.0    # Normalize ab to [-1, 1]

        L = lab[:, :, 0:1]  # Extract grayscale (L channel)
        ab = lab[:, :, 1:]  # Extract color channels (a and b)
        
        L_tensor = torch.from_numpy(L).permute(2, 0, 1).float()  # Shape: [1, H, W]
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1).float()  # Shape: [2, H, W]

        return L_tensor, ab_tensor

# ==== TEST, EVALUATE, AND SAVE MODEL IMAGES ====
def evaluate_and_save(model, test_loader, device, output_folder="PredictedColorizedImg"):
    
    # Remove existing files in the output directory
    if os.path.exists(output_folder):
        files = glob.glob(os.path.join(output_folder, '*'))
        for f in files:
            os.remove(f)
            
    # Make the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Evaluate the model
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for i, (L_batch, ab_batch) in enumerate(test_loader):
            L_batch = L_batch.to(device)
            ab_batch = ab_batch.to(device)
            preds = model(L_batch)
            loss = nn.functional.mse_loss(preds, ab_batch, reduction='mean')
            total_loss += loss.item()
            count += 1

            # Save RGB colorized results and calculate PSNR/SSIM
            for j in range(L_batch.size(0)):
                # Convert LAB to RGB for ground truth and predictions
                L = L_batch[j].cpu().numpy().squeeze() * 100
                ab_pred = preds[j].cpu().numpy().transpose(1, 2, 0) * 128
                ab_gt = ab_batch[j].cpu().numpy().transpose(1, 2, 0) * 128

                # Clamp predicted ab values to the valid range
                ab_pred = np.clip(ab_pred, -128, 127)
                ab_gt = np.clip(ab_gt, -128, 127)

                lab_pred = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
                lab_gt = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)

                lab_pred[:, :, 0] = L
                lab_pred[:, :, 1:] = ab_pred
                lab_gt[:, :, 0] = L
                lab_gt[:, :, 1:] = ab_gt

                rgb_pred = np.clip(lab2rgb(lab_pred), 0, 1)  # Ensure values are in [0, 1]
                rgb_gt = np.clip(lab2rgb(lab_gt), 0, 1)      # Ensure values are in [0, 1]

                # Sanity check for normalization
                assert np.max(rgb_pred) <= 1.0 and np.min(rgb_pred) >= 0.0, "rgb_pred is not normalized!"
                assert np.max(rgb_gt) <= 1.0 and np.min(rgb_gt) >= 0.0, "rgb_gt is not normalized!"

                # Save predicted image
                rgb_img = (rgb_pred * 255).astype(np.uint8)
                img = Image.fromarray(rgb_img)
                img.save(os.path.join(output_folder, f"test_img_{i*10 + j}.png"))

                # Calculate PSNR and SSIM
                psnr_value = psnr(rgb_gt, rgb_pred, data_range=1.0)  # Use 1.0 for normalized images
                ssim_value = ssim(rgb_gt, rgb_pred, win_size=5, channel_axis=-1, data_range=1.0)

                if psnr_value > 100 or ssim_value > 1:  # Unusually high values
                    print(f"Debugging image {i*10 + j}")
                    Image.fromarray((rgb_gt * 255).astype(np.uint8)).save(f"debug_gt_{i*10 + j}.png")
                    Image.fromarray((rgb_pred * 255).astype(np.uint8)).save(f"debug_pred_{i*10 + j}.png")
                    np.save(f"debug_lab_gt_{i*10 + j}.npy", lab_gt)  # Save LAB ground truth
                    np.save(f"debug_lab_pred_{i*10 + j}.npy", lab_pred)  # Save LAB prediction

                total_psnr += psnr_value
                total_ssim += ssim_value

    # Calculate average metrics
    mse = total_loss / count
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    # Print out evaluation metrics
    print(f"Test MSE: {mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

# ==== GENERATE AUGMENTED IMAGES ====
def generate_augmented_images(img_to_aug, output_dir="augmented"):
    # Ensure the augmented directory is in the same directory as this script
    output_dir = os.path.join(os.getcwd(), output_dir)

    # Remove existing files in the augmented directory
    if os.path.exists(output_dir):
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)

    # Create the augmented directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read in every image and resize them to 128 by 128
    original_images = []
    for f in img_to_aug:
        img = cv2.imread(f)  # BGR read in
        if img is None:  # Skip unreadable files
            print(f"Warning: Unable to read image {f}. Skipping.")
            continue
        img = cv2.resize(img, (128, 128))
        original_images.append(img)

    original_images = np.array(original_images)
    print(f"Loaded {len(original_images)} images")

    # Data augmentation factor
    AUG_FACTOR = 5  # Number of augmentations per image
    augmented_images = []

    def augment_image(img):
        # Random flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # Random crop and resize
        h, w = img.shape[:2]
        crop_size = random.randint(int(0.7 * h), h)  # Reduce crop size to make it more challenging
        y = random.randint(0, h - crop_size)
        x = random.randint(0, w - crop_size)
        cropped = img[y:y+crop_size, x:x+crop_size]
        cropped = cv2.resize(cropped, (128, 128))

        # Random brightness scale
        scale = random.uniform(0.5, 1.2)  # Increase brightness variation
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
            cv2.imwrite(os.path.join(output_dir, filename), aug_img)

            counter += 1

    print(f"Generated and saved {counter} augmented images.")

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

def transfer_learning_on_ncdataset(model, device, color_dir, num_epochs=10, batch_size=10, initial_lr=1e-4, unfreeze_interval=3):
    # Fetch all image paths from the ColorfulOriginal dataset
    color_image_paths = get_image_paths_from_directory(color_dir)

    # Generate augmented images in the same directory as this script
    augmented_dir = os.path.join(os.getcwd(), "augmented")
    generate_augmented_images(color_image_paths, output_dir=augmented_dir)

    # Combine original and augmented image paths
    augmented_image_paths = get_image_paths_from_directory(augmented_dir)
    all_image_paths = color_image_paths + augmented_image_paths

    # Split into training (80%), validation (10%), and testing (10%) datasets
    train_size = int(0.8 * len(all_image_paths))
    val_size = int(0.1 * len(all_image_paths))
    test_size = len(all_image_paths) - train_size - val_size
    train_color, val_color, test_color = random_split(all_image_paths, [train_size, val_size, test_size])

    # Create datasets and dataloaders
    train_dataset = ColorfulOriginalDataset(train_color, augment=True)
    val_dataset = ColorfulOriginalDataset(val_color, augment=False)
    test_dataset = ColorfulOriginalDataset(test_color, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Fine-tune the model
    model.to(device)
    num_layers_to_unfreeze = 4  # Start with the first 4 layers frozen

    for epoch in range(num_epochs):
        # Unfreeze additional layers at specified intervals
        if epoch > 0 and epoch % unfreeze_interval == 0:
            num_layers_to_unfreeze += 1
            unfreeze_layers(model, num_layers_to_unfreeze)
            print(f"Epoch {epoch}: Unfroze {num_layers_to_unfreeze} layers.")

        total_train_loss = 0
        total_val_loss = 0

        # Training
        model.train()
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            preds = model(L_batch)
            loss = criterion(preds, ab_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for L_batch, ab_batch in val_loader:
                L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
                preds = model(L_batch)
                loss = criterion(preds, ab_batch)
                total_val_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_train_loss/len(train_loader):.4f}, Validation Loss = {total_val_loss/len(val_loader):.4f}")

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
            
def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreeze the first `num_layers_to_unfreeze` layers of the model.
    """
    layers = list(model.children())  # Get all layers of the model
    for layer in layers[:num_layers_to_unfreeze]:
        for param in layer.parameters():
            param.requires_grad = True


# ==== MAIN ====
def main():
    base_dir = os.getcwd()
    color_dir = os.path.join(base_dir, "ColorfulOriginal")

    # Load the pretrained model
    model_path = os.path.join(base_dir, "colorization_model_face.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationCNN()
    model = load_pretrained_model(model, model_path, device)

    # Freeze the first 4 layers of the model
    freeze_layers(model, num_layers_to_freeze=4)

    # Perform transfer learning on the ColorfulOriginal dataset
    transfer_learning_on_ncdataset(model, device, color_dir)

if __name__ == "__main__":
    main()