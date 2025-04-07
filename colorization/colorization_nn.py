#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 20:26:52 2025

@author: mahdiali-raihan
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F

class ColorizationCNN(nn.Module):
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        
        # Downsampling with Convolution, Batch Normalization
        self.down1 = self.upconv_layer(1, 64, 3, 1, 1)
        self.down2 = self.upconv_layer(64, 128)
        self.down3 = self.upconv_layer(128, 256)
        self.down4 = self.upconv_layer(256, 512)
        self.down5 = self.upconv_layer(512, 1024)
        self.down6 = self.upconv_layer(1024, 2048)
        
        # self.up5 = self.downconv_layer(512, 512)
        # self.up4 = self.downconv_layer(512, 256)
        # self.up3 = self.downconv_layer(256, 128)
        # self.up2 = self.downconv_layer(128, 64)
        # self.up1 = self.downconv_layer(64, 2)  # Output 2 channels for a* and b*
        self.up6 = self.downconv_layer(2048, 1024)
        self.up5 = self.downconv_layer(1024, 512)
        self.up4 = self.downconv_layer(512, 256)
        self.up3 = self.downconv_layer(256, 128)
        self.up2 = self.downconv_layer(128, 64)
        self.up1 = self.downconv_layer(64, 2, 3)

        
        
    
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
        x = self.down6(x)

        # Performing upsampling
        x = self.up6(x)
        x = self.up5(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        
        return x
