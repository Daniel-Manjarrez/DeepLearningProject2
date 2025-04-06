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
    
        # Encoder ()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 640x480
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),

        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),

        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )

        # # Decoder (no upsampling needed since spatial size is constant)
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),

        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),

        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),

        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
        #     nn.Tanh(),  # Output: a*, b*
        # )
        
        # Downsampling with Convolution, Batch Normalization
        self.down1 = self.conv_block(1, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 512)
        
        self.up5 = self.deconv_block(512, 512)
        self.up4 = self.deconv_block(512, 256)
        self.up3 = self.deconv_block(256, 128)
        self.up2 = self.deconv_block(128, 64)
        self.up1 = self.deconv_block(64, 2)  # Output 2 channels for a* and b*
        
        
    
    def upconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def downconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
