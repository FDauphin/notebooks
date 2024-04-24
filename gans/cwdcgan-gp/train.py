"""
Train a Conditional Wasserstein DCGAN using MNIST data and gradient penalty.

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
- use WEIGHT_CLIP = 0.01 to clip model parametes to that boundary
- use a learning rate of 5e-5
- use RMSProp optimizer
- remove gradient penalty portion of loss in training loop
- uncomment weight clipping code snippet

Saves various loss metrics as a .csv file and model parameters as .pt files.

This script is intended to run via command line:

    >>> python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from model import Critic, Generator, gradient_penalty

# Prepare transformations
CHANNELS_IMG = 1
means = [0.5 for _ in range(CHANNELS_IMG)]
stds = [0.5 for _ in range(CHANNELS_IMG)]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])

# Load data with transformations
root = '../'
dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
IMAGE_SIZE = dataset.data.shape[-1]
NUM_CLASSES = np.unique(dataset.targets).shape[0]

# Prepare dataset
BATCH_SIZE = 128 # original paper uses 64
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Load models
FEATURES = 32
NOISE_DIM = 100

critic = Critic(features=FEATURES, channels_img=CHANNELS_IMG, img_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
gen = Generator(channels_noise=NOISE_DIM, features=FEATURES, channels_img=CHANNELS_IMG, num_classes=NUM_CLASSES)

# Put models on device (mps for Mac M1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
critic.to(device)
gen.to(device)

# Define gradient penalty parameters
LAMBDA_GP = 10
#WEIGHT_CLIP = 0.01 # for wgan without gradient penalty

# Define optimization method
LEARNING_RATE = 1e-4 # 5e-5 for wgan without gradient penalty
BETAS = (0.0, 0.9)
#RMSProp and no beta for wgan without gradient penalty
opt_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=BETAS)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)

# Train models
loss = []
NUM_EPOCHS = 25
CRITIC_ITERATIONS = 5
for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
    critic.train()
    gen.train()
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.float().to(device)
        labels = labels.type(torch.LongTensor).to(device)
        
        # Train Critic: min -(E[critic(real)] - E[critic(fake)])
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(real.shape[0], NOISE_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            loss_critic_real = torch.mean(critic_real)
            critic_fake = critic(fake, labels).reshape(-1)
            loss_critic_fake = torch.mean(critic_fake)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic_gp = LAMBDA_GP * gp
            loss_critic = loss_critic_fake - loss_critic_real + loss_critic_gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # for wgan without gradient penalty
            #for p in critic.parameters():
                #p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                
        # Train Generator: min -E[critic(fake)]
        output = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    loss.append([loss_critic_real.item(), loss_critic_fake.item(), loss_critic_gp.item(), loss_gen.item()])

# Save loss to dataframe
now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
loss = np.array(loss)
columns = ['Loss Critic Real', 'Loss Critic Fake', 'Loss Critic Gradient Penalty', 'Loss Gen']
df = pd.DataFrame(loss, columns=columns)
df.to_csv(f'cwdcgan-gp_loss_{now}.csv')

# Save models' state dicts
critic.eval()
gen.eval()
torch.save(critic.state_dict(), f'cwdcgan-gp_critic_{now}.pt')
torch.save(gen.state_dict(), f'cwdcgan-gp_gen_{now}.pt')