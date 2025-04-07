import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

# Set default tensor type
torch.set_default_dtype(torch.float32)

# ==== COLORIZATION CNN ====
class ColorizationCNN(nn.Module):
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        
        # Downsampling with Convolution, Batch Normalization
        # self.down1 = self.upconv_layer(1, 64, 3, 1, 1)
        # self.down2 = self.upconv_layer(64, 128)
        # self.down3 = self.upconv_layer(128, 256)
        # self.down4 = self.upconv_layer(256, 512)
        # self.down5 = self.upconv_layer(512, 1024)
        # self.down6 = self.upconv_layer(1024, 2048)

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
        
        # self.up5 = self.downconv_layer(512, 512)
        # self.up4 = self.downconv_layer(512, 256)
        # self.up3 = self.downconv_layer(256, 128)
        # self.up2 = self.downconv_layer(128, 64)
        # self.up1 = self.downconv_layer(64, 2)  # Output 2 channels for a* and b*
        # self.up6 = self.downconv_layer(2048, 1024)
        # self.up5 = self.downconv_layer(1024, 512)
        # self.up4 = self.downconv_layer(512, 256)
        # self.up3 = self.downconv_layer(256, 128)
        # self.up2 = self.downconv_layer(128, 64)
        # self.up1 = self.downconv_layer(64, 2, ksize=3, stride = 1, padding=1)
    
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
        # x = self.down6(x)

        # Performing upsampling
        # x = self.up6(x)
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
        # # img = cv2.imread(self.image_paths[idx])
        # img = Image.open(self.image_paths[idx]).convert("RGB")
        # img = cv2.resize(img, (128, 128))  # Resize all to 128x128
        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # L, a, b = cv2.split(lab)
        # L = L.astype(np.float32) / 100.0  # Normalize to [0, 1]

        # # a_mean = a.mean() - 128.0  # Center a*
        # # b_mean = b.mean() - 128.0  # Center b*
        # L_tensor = torch.tensor(L).unsqueeze(0)  # Shape: (1, H, W)
        # target = torch.tensor([a_mean, b_mean], dtype=torch.float32)
        # return L_tensor, target
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
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, (L_batch, ab_batch) in enumerate(test_loader):
            L_batch = L_batch.to(device)
            ab_batch = ab_batch.to(device)
            preds = model(L_batch)
            loss = nn.functional.mse_loss(preds, ab_batch, reduction='sum')
            total_loss += loss.item()
            count += L_batch.size(0)

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

    mse = total_loss / (count * 2 * 640 * 480)
    print(f"Test MSE: {mse:.6f}")


# ==== MAIN ====
def main():
    # Get the current working directory (where the script is being run from)
    base_dir = os.getcwd()

    # Construct the path to the 'augmented' directory
    aug_dir = os.path.join(base_dir, "augmented")
    face_dir = os.path.join(base_dir, "face_images")

    # Check if the augmented directory exists
    if not os.path.exists(aug_dir):
        print(f"Error: The directory '{aug_dir}' does not exist.")
        return
    
    aug_paths = [os.path.join(aug_dir, fname)
                            for fname in os.listdir(aug_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    train_aug_paths = [aug_paths[x] for x in range(int(len(aug_paths) * 0.9))]

    train_dataset = ColorizationDataset(train_aug_paths)


    face_paths = [os.path.join(face_dir, fname)
                            for fname in os.listdir(face_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    test_face_paths = [face_paths[int(len(face_paths) * 0.9) - 1 + x] for x in range(int(len(face_paths) * 0.1))]

    test_dataset = ColorizationDataset(test_face_paths)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Grab custom made Colorization CNN model
    model = ColorizationCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Determine whether we're running the model on cuda or cpu 
    # (for now just do cpu)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    
    total_loss = 0
    num_epochs = 10

    # Running with 10 epochs 
    for epoch in range(num_epochs):
        # Train model first
        model.train()
        
        # For each L_batch, ab_batch from the train_loader
        # Check how accurate the images are to the colored one
        # for L_batch, ab_batch in train_loader:
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
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss/len(train_loader):.4f}")

    
    # After training, save predictions
    evaluate_and_save(model, test_loader, device=device)

if __name__ == "__main__":
    main()
