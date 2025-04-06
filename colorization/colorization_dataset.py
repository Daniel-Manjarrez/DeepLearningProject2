#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:08:15 2025

@author: mahdiali-raihan
"""

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, gray_paths, color_paths, augment=False):
        self.gray_paths = gray_paths
        self.color_paths = color_paths
        self.augment = augment

        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])

    def __len__(self):
        return len(self.gray_paths)

    def __getitem__(self, idx):
        gray_img_path = self.gray_paths[idx]
        color_img_path = self.color_paths[idx]

        gray = Image.open(gray_img_path).convert("L").resize((480, 640))
        color = Image.open(color_img_path).convert("RGB").resize((480, 640))

        if self.augment:
            seed = np.random.randint(9999)
            gray = transforms.functional.pil_to_tensor(gray).float() / 255
            color = transforms.functional.pil_to_tensor(color).float() / 255

            torch.manual_seed(seed)
            gray = self.augment_transform(gray)

            torch.manual_seed(seed)
            color = self.augment_transform(color)

            gray = transforms.ToPILImage()(gray)
            color = transforms.ToPILImage()(color)

        # Convert to Lab
        gray_np = np.asarray(gray) / 255.0
        color_np = np.asarray(color) / 255.0
        lab = rgb2lab(color_np).astype("float32")
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:] / 128.0

        L_tensor = torch.tensor(L).unsqueeze(0)
        ab_tensor = torch.tensor(ab).permute(2, 0, 1)

        return L_tensor, ab_tensor
