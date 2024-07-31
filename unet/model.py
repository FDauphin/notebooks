"""
UNET model for image to image tasks

Paper: https://arxiv.org/abs/1505.04597

This module is intended to be imported in Python:

    >>> from model import UNET

Notes
-----
- invariant to conv vs fc (with or w/o relu)
- skip connections significantly improve performance
try:
- downsample with just convolutions
- upsample with bicubic interpolation then conv
"""

import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self,
                 n_channels=1,
                 filts=32,
                 kernels=[3, 4],
                 strides=[1, 2],
                 paddings=[1, 1],
                 pool=2,
                 block5_type='conv'):
        super(UNET, self).__init__()

        # Hyperparameters
        self.n_channels = n_channels
        self.filt1 = filts
        self.filt2 = filts * 2
        self.filt3 = filts * 4
        self.filt4 = filts * 8
        self.filt5 = filts * 16
        
        self.kernel_down = kernels[0]
        self.kernel_up = kernels[1]
        self.stride_down = strides[0]
        self.stride_up = strides[1]
        self.padding_down = paddings[0]
        self.padding_up = paddings[1]
        self.pool = pool
        self.block5_type = block5_type.lower()

        # Functions
        self.mp = nn.MaxPool2d(self.pool)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (self.filt5, 1, 1))
        self.sigmoid = nn.Sigmoid()

        # Blocks: down sample
        self.block1 = self._block_down(self.n_channels, self.filt1, self.kernel_down, self.stride_down, self.padding_down)
        self.block2 = self._block_down(self.filt1, self.filt2, self.kernel_down, self.stride_down, self.padding_down)
        self.block3 = self._block_down(self.filt2, self.filt3, self.kernel_down, self.stride_down, self.padding_down)
        self.block4 = self._block_down(self.filt3, self.filt4, self.kernel_down, self.stride_down, self.padding_down)

        # Blocks: bottleneck
        if self.block5_type == 'conv':
            self.block5 = nn.Sequential(
                nn.Conv2d(self.filt4, self.filt5, self.kernel_down, self.stride_down, self.padding_down, bias=True),
                self.relu
            )
            self.block6 = self._block_up(self.filt5, self.filt4, self.kernel_up, self.stride_up, self.padding_up, 
                                         output_padding=self.padding_up)
        elif self.block5_type == 'fc':
            self.block5 = nn.Sequential(
                self.flatten,
                nn.Linear(self.filt4, 2, bias=True),
                self.relu,
            )
            self.block6 = nn.Sequential(
                nn.Linear(2, self.filt5, bias=True),
                self.unflatten,
                self._block_up(self.filt5, self.filt4, self.kernel_up, self.stride_up, self.padding_up, 
                               output_padding=self.padding_up)
            )
        else:
            raise ValueError(f"{self.block5_type} is not an accepted value for block5_type; use 'conv' or 'fc'.")
            
        # Blocks: up sample
        self.block7 = self._block_up(self.filt5, self.filt3, self.kernel_up, self.stride_up, self.padding_up, 
                                     output_padding=self.padding_up)
        self.block8 = self._block_up(self.filt4, self.filt2, self.kernel_up, self.stride_up, self.padding_up, 
                                     output_padding=0)
        self.block9 = self._block_up(self.filt3, self.filt1, self.kernel_up, self.stride_up, self.padding_up, 
                                     output_padding=0)
        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(self.filt2, self.n_channels, self.kernel_down, self.stride_down, self.padding_down, bias=True),
            self.sigmoid
        )

    # Blocks: functions
    def _block_down(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
            )

    def _block_up(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            self.relu
            )

    # Blocks: down and up
    def down(self, x):
        feature_maps = []
        blocks_down = nn.ModuleList([self.block1, self.block2, self.block3, self.block4])
        for block in blocks_down:
            x = block(x)
            feature_maps.append(x)
            x = self.mp(x)
        x = self.block5(x)
        return x, feature_maps

    def up(self, x, feature_maps):
        blocks_up = nn.ModuleList([self.block6, self.block7, self.block8, self.block9])
        for i, block in enumerate(blocks_up):
            x = block(x)
            x = torch.cat([x, feature_maps[-(i+1)]], dim=1)
        x = self.block10(x)
        return x

    # Forward pass
    def forward(self, x):
        x, feature_maps = self.down(x)
        x = self.up(x, feature_maps)
        return x