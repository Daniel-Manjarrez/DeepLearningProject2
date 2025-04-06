import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Set default tensor type
torch.set_default_dtype(torch.float32)

# ==== CNN REGRESSOR ====
class CNNRegressor(nn.Module):
    def __init__(self, in_channels=1):
        super(CNNRegressor, self).__init__()
        layers = []
        channels = in_channels
        out_channels = 3
        for _ in range(7):
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
        self.conv_stack = nn.Sequential(*layers)
        self.fc = nn.Linear(out_channels, 2)

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc(x)
        return x

# ==== DATASET ====
class LABMeanDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (128, 128))  # Resize all to 128x128
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        L = L.astype(np.float32) / 100.0  # Normalize to [0, 1]
        a_mean = a.mean() - 128.0  # Center a*
        b_mean = b.mean() - 128.0  # Center b*
        L_tensor = torch.tensor(L).unsqueeze(0)  # Shape: (1, H, W)
        target = torch.tensor([a_mean, b_mean], dtype=torch.float32)
        return L_tensor, target

# ==== TRAINING FUNCTION ====
def train(model, train_dataloader, val_dataloader, device, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')  # Initialize best loss as infinity
    best_model_state_dict = None  # To store the best model's state dict

    for epoch in range(epochs):
        total_loss = 0
        # Training loop
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation loop (calculate validation loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                loss = loss_fn(preds, targets)
                val_loss += loss.item()
        
        # Print training and validation loss for the epoch
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {total_loss:.4f} - Validation Loss: {val_loss:.4f}")
        
        # Check if this is the best validation loss we've seen so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()  # Save the best model's state

        model.train()  # Switch back to training mode
        
    # Save the best model after training
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        torch.save(model.state_dict(), "best_cnn_regressor.pth")
        print("Best model saved as best_cnn_regressor.pth")

# ==== MAIN ====
def main():
    # Get the current working directory (where the script is being run from)
    base_dir = os.getcwd()

    # Construct the path to the 'augmented' directory
    image_dir = os.path.join(base_dir, "augmented")

    # Check if the augmented directory exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory '{image_dir}' does not exist.")
        return

    dataset = LABMeanDataset(image_dir)
    
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRegressor().to(device)

    print("Starting training...")
    train(model, train_dataloader, val_dataloader, device, epochs=10)

if __name__ == "__main__":
    main()