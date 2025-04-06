#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:08:15 2025

@author: mahdiali-raihan
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.model_selection import train_test_split

class ColorizationDataset(Dataset):
    def __init__(self, color_image_paths, transform=None):
        self.color_paths = color_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        img = Image.open(self.color_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        lab = rgb2lab(np.array(img)).astype("float32")
        lab[:, :, 0] = lab[:, :, 0] / 100.0      # Normalize L to [0, 1]
        lab[:, :, 1:] = lab[:, :, 1:] / 128.0    # Normalize ab to [-1, 1]

        L = lab[:, :, 0:1]
        ab = lab[:, :, 1:]

        L_tensor = torch.from_numpy(L).permute(2, 0, 1)  # Shape: [1, H, W]
        ab_tensor = torch.from_numpy(ab).permute(2, 0, 1)  # Shape: [2, H, W]

        return L_tensor, ab_tensor