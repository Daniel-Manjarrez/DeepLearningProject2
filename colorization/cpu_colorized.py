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
import time

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
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        lab = rgb2lab(np.array(img)).astype("float32")
        lab[:, :, 0] /= 100.0      # Normalize L to [0, 1]
        lab[:, :, 1:] /= 128.0    # Normalize ab to [-1, 1]

        L = lab[:, :, 0:1]
        ab = lab[:, :, 1:]

        L_tensor = torch.from_numpy(L).permute(2, 0, 1)  # Shape: [1, H, W]
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)  # Shape: [2, H, W]

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



# ==== MAIN ====
def main():
    # Get the current working directory (where the script is being run from)
    base_dir = os.getcwd()

    # Construct the path to the 'face_images' directory
    face_dir = os.path.join(base_dir, "face_images")

    # Fetch all image filenames from the face_images folder
    face_paths = [os.path.join(face_dir, fname)
                            for fname in os.listdir(face_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split 90% of the face images to augment, 10% to remain the same
    train_size = 0.9
    test_size = 0.1
    to_augment, to_original = random_split(face_paths, [train_size, test_size])

    # Construct path to augmented directory
    aug_dir = os.path.join(base_dir, "augmented/")

    # Delete all files inside the augmented directory if there are any
    if os.path.exists(aug_dir):
        files = glob.glob(os.path.join(aug_dir, '*'))
        for f in files:
            os.remove(f)

    # Generated new augmented images
    generate_augmented_images(to_augment)

    # Fetch all image file names in the augmented path
    aug_paths = [os.path.join(aug_dir, fname)
                            for fname in os.listdir(aug_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Kinda redundant, but just changed variable name to store it in
    train_aug_paths = aug_paths

    # Grab the L*, ab* colorized dataset
    train_dataset = ColorizationDataset(train_aug_paths)
    
    # Redundant again, but keep the last 10% original into the test dataset
    test_face_paths = to_original

    # Grab the L*, ab* of the actual face images
    test_dataset = ColorizationDataset(test_face_paths)

    # DataLoad both the training and testing data set
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Grab custom made Colorization CNN model
    model = ColorizationCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Determine whether we're running the model on cuda or cpu 
    device = torch.device("cpu")
    model.to(device)
    
    train_start_time = time.time()
    
    # Running with 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        # Train model first
        model.train()
        
        # For each L_batch, ab_batch from the train_loader
        # Check how accurate the images are to the colored one
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            # clears gradient from prev step
            optimizer.zero_grad()

            # make prediction
            preds = model(L_batch)

            # Calculate loss
            loss = criterion(preds, ab_batch)

            # perform backpropagation
            loss.backward()

            # Update model's parameters 
            optimizer.step()

            total_loss += loss.item()
        
        # Print out epoch results
        # print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss/loss.item():.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}")
        # print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss:.4f}")
        
    train_end_time = time.time()
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")
    # Write the runtime to a file
    output_file = "cpu_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"Training time: {train_end_time - train_start_time:.2f} seconds", file=f)
        print(f"---------------------------------------", file=f)
    
    # Construct path to the predicted colorized img directory
    predict_dir = os.path.join(base_dir, "PredictedColorizedImg/")
    # Create folders if they do not exist
    os.makedirs(predict_dir, exist_ok=True)

    # Clear all files inside this folder from previous run
    if os.path.exists(predict_dir):
        files = glob.glob(os.path.join(predict_dir, '*'))
        for f in files:
            os.remove(f)

    # After training, save predictions
    evaluate_and_save(model, test_loader, device=device)

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    
    # Call the main function
    main()
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the total runtime
    total_time = end_time - start_time
    
    # Write the runtime to a file
    output_file = "cpu_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"---------------------------------------", file=f)
        print(f"Total runtime: {total_time:.2f} seconds", file=f)
    
    # Print to console total runtime too 
    print(f"Total runtime: {total_time:.2f} seconds")
