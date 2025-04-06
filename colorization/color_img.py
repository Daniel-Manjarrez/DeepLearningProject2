#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 20:48:03 2025

@author: mahdiali-raihan
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from colorization_dataset import ColorizationDataset
from colorization_nn import ColorizationCNN

import os
from skimage.color import lab2rgb
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split


# Grabbing all subfolders from "Gray" and "ColorOriginal"
def collect_image_paths(gray_dir, color_dir):
    gray_paths = []
    color_paths = []

    for subdir in os.listdir(gray_dir):
        gray_subdir = os.path.join(gray_dir, subdir)
        color_subdir = os.path.join(color_dir, subdir)
        if not os.path.isdir(gray_subdir): continue

        for fname in os.listdir(gray_subdir):
            g = os.path.join(gray_subdir, fname)
            c = os.path.join(color_subdir, fname)
            if os.path.exists(g) and os.path.exists(c):
                gray_paths.append(g)
                color_paths.append(c)
    
    return gray_paths, color_paths

# Evaluate the model results and save it to "PredictedColorizedImg"
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



if __name__ == "__main__":
    
    # The folders where the grayscaled and colored images are located at
    gray_dir = "Gray"
    color_dir = "ColorfulOriginal"
    # All subfolders inside
    gray_paths, color_paths = collect_image_paths(gray_dir, color_dir)
    
    # Train/Test split, 90% training, 10% testing
    train_gray, test_gray, train_color, test_color = train_test_split(
        gray_paths, color_paths, test_size=0.1, random_state=42
    )
    
    # ColorizationDataset used to fetch images and convert to L, a*, b*
    train_dataset = ColorizationDataset(train_gray, train_color, augment=True)
    test_dataset = ColorizationDataset(test_gray, test_color, augment=False)
    
    # Dataloader running 10 minibatches fortraining/testing (shuffle for training)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
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
    
    
    # Running with 10 epochs 
    for epoch in range(10):
        # Train model first
        model.train()
        
        # For each L_batch, ab_batch from the train_loader
        # Check how accurate the images are to the colored one
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)
            preds = model(L_batch)
            loss = criterion(preds, ab_batch)
            
            # clears gradient from prev step
            optimizer.zero_grad()
            # perform backpropagation
            loss.backward()
            # Update model's parameters 
            optimizer.step()
        
        # Print out epoch results
        print(f"Epoch {epoch+1}: Training Loss = {loss.item():.4f}")
    
    # After training, save predictions
    evaluate_and_save(model, test_loader, device=device)

    
    
    
    
    
    