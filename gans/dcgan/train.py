"""
Train a DCGAN using MNIST data.

DCGAN paper:
https://arxiv.org/abs/1511.06434

Most of source code from:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/train.py

The script uses just the scaled tensors for the DataLoader instead of using the 
default torch object. This method worked best for my resources to produce the 
fastest results. I used a 2021 Mac M1 with my environment in the repository 
built on the 2020 MacOSX Intel Miniconda.

1. Training on the torch object when device=cpu uses all cpus in >>>60 minutes (too slow).
2. Training on the torch object when device=mps uses one cpu and the gpu in ~38 minutes.
3. Training on the tensors when device=mps uses one cpu and the gpu in ~30 minutes (best).
4. Training on a numpy array uses all cpus and the gpu (when device=mps) to full efficiency in ~30 minutes.

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

from model import Discriminator, Generator

# Prepare transformations
CHANNELS_IMG = 1
means = [0.5 for _ in range(CHANNELS_IMG)]
stds = [0.5 for _ in range(CHANNELS_IMG)]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])

# Load data with transformations
root = '../'
dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
IMAGE_SIZE = dataset.data.shape[-1]

# Manually scale dataset with image channel
scale = dataset.data.max()
x_train = ((dataset.data / scale) - 0.5) / 0.5
dataset = x_train.reshape(x_train.shape[0], CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)

# Prepare dataset
BATCH_SIZE = 128
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
# Load models
FEATURES = 32
NOISE_DIM = 100

disc = Discriminator(features=FEATURES, channels_img=CHANNELS_IMG)
gen = Generator(channels_noise=NOISE_DIM, features=FEATURES, channels_img=CHANNELS_IMG)

# Put models on device (mps for Mac M1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
disc.to(device)
gen.to(device)

# Define loss function
criterion = nn.BCELoss()

# Define optimization method
LEARNING_RATE = 3e-4 # original dcgan paper uses 2e-4
BETAS = (0.9, 0.999) # original dcgan paper uses 0.5 and 0.999
opt_disc = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)

# Train models
loss = []
NUM_EPOCHS = 100
for epoch in tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS):
    disc.train()
    gen.train()
    for batch_idx, real in enumerate(dataloader):
        real = real.float().to(device)
        noise = torch.randn(real.shape[0], NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Append loss
    loss.append([loss_disc_real.item(), loss_disc_fake.item(), disc_real.mean().item(), disc_fake.mean().item(),
                 loss_gen.item(), output.mean().item()])

# Save loss to dataframe
now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
loss = np.array(loss)
columns = ['Loss Disc Real', 'Loss Disc Fake', 'Mean Disc Real', 'Mean Disc Fake', 'Loss Gen', 'Mean Disc Fake (Gen Training)']
df = pd.DataFrame(loss, columns=columns)
df.to_csv(f'dcgan_loss_{now}.csv')

# Save models' state dicts
disc.eval()
gen.eval()
torch.save(disc.state_dict(), f'dcgan_disc_{now}.pt')
torch.save(gen.state_dict(), f'dcgan_gen_{now}.pt')
