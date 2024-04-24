"""
Critic and generator models for CWDGCAN. Also includes function for gradient penalty.

WGAN paper:
https://arxiv.org/abs/1701.07875

WGAN-GP paper:
https://arxiv.org/abs/1704.00028

CGAN paper:
https://arxiv.org/abs/1411.1784

Most of source code from:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/train.py

I used a 2021 Mac M1 with my environment in the repository built on the 2020 
MacOSX Intel Miniconda. Training took approximately one hour.

To implement without gradient penalty:
- use BatchNorm2d instead of InstanceNorm2d in the critic

This module is intended to be imported in Python:

    >>> from model import Critic, Generator, gradient_penalty
"""

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

class Critic(nn.Module):
    def __init__(self,
                 features,
                 channels_img,
                 img_size,
                 num_classes,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 alpha=0.2):
        super(Critic, self).__init__()

        # Hyperparameters
        self.features = features
        self.channels_img = channels_img
        self.img_size = img_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        
        # Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(self.channels_img + 1, self.features, 
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(self.alpha)
        )
        self.block2 = self._block(self.features, self.features * 2,
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.block3 = self._block(self.features * 2, self.features * 4, 
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.block4 = nn.Conv2d(self.features * 4, self.channels_img, 
                        kernel_size=self.kernel_size-1, stride=self.stride, padding=self.padding-1)

        # Network
        self.critic = nn.Sequential(# 1 x 28 x 28
            self.block1, # 32 x 14 x 14
            self.block2, # 64 x 7 x 7
            self.block3, # 128 x 3 x 3
            self.block4  # 1 x 1 x 1
        )

        # Embedding (acts as an extra channel)
        self.embed = nn.Embedding(self.num_classes, self.img_size ** 2)
            
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # batchnorm2d for wgan without gp penalty
            # layernorm ~ similar norm (doesn't normalize across batches)
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(self.alpha),
        )

    def _cat_embedding(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x [C + 1 (channel for label embedding)] x H x W
        return x

    def forward(self, x, labels):
        x = self._cat_embedding(x, labels)
        return self.critic(x)

class Generator(nn.Module):
    def __init__(self,
                 channels_noise,
                 features,
                 channels_img,
                 num_classes,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(Generator, self).__init__()

        # Hyperparameters
        self.channels_noise = channels_noise
        self.features = features
        self.channels_img = channels_img
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embed_size = self.channels_noise

        # Blocks
        self.block1 = self._block(self.channels_noise + self.embed_size, self.features * 4, 
                        kernel_size=self.kernel_size-1, stride=self.stride, padding=self.padding-1)
        self.block2 = self._block(self.features * 4, self.features * 2,
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                        output_padding=1)
        self.block3 = self._block(self.features * 2, self.features, 
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(self.features, self.channels_img, 
                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Tanh()
        )

        # Network
        self.gen = nn.Sequential(# channel_noise x 1 x 1
            self.block1, # 128 x 3 x 3
            self.block2, # 64 x 7 x 7
            self.block3, # 8 x 14 x 14
            self.block4  # 1 x 28 x 28
        )

        # Embedding (acts as an extra channel)
        self.embed = nn.Embedding(self.num_classes, self.embed_size)
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _cat_embedding(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1) # N x (noise_dim + embed_size) x 1 x 1
        return x

    def forward(self, x, labels):
        x = self._cat_embedding(x, labels)
        return self.gen(x)

def gradient_penalty(critic, labels, real, fake, device='mps'):
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    
    # calculate critic scores
    mixed_scores = critic(interpolated_images, labels)
    gradient = torch.autograd.grad(inputs=interpolated_images,
                                   outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True,
                                   retain_graph=True)[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
    