"""
Discriminator and generator models for a DCGAN.

Also contains a function to invert real data to the latent space.

DCGAN paper:
https://arxiv.org/abs/1511.06434

GAN inversion survey:
https://arxiv.org/abs/2101.05278

Model source code from:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/model.py

Inversion source code from:
https://github.com/genforce/idinvert_pytorch/blob/master/utils/inverter.py

This module is intended to be imported in Python:

    >>> from model import Discriminator, Generator, invert
"""

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, 
                 features, 
                 channels_img,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 alpha=0.2):
        super(Discriminator, self).__init__()

        # Hyperparameters
        self.features = features
        self.channels_img = channels_img
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha

        # Blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(self.channels_img, self.features, 
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(self.alpha)
        )
        self.block2 = self._block(self.features, self.features * 2,
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.block3 = self._block(self.features * 2, self.features * 4, 
                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.block4 = nn.Sequential(
            nn.Conv2d(self.features * 4, self.channels_img, 
                      kernel_size=self.kernel_size-1, stride=self.stride, padding=self.padding-1),
            nn.Sigmoid()
        )

        # Network
        self.disc = nn.Sequential(# 1 x 28 x 28
            self.block1, # 32 x 14 x 14
            self.block2, # 64 x 7 x 7
            self.block3, # 128 x 3 x 3
            self.block4  # 1 x 1 x 1
        )
            
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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(self.alpha),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, 
                 channels_noise,
                 features,
                 channels_img,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(Generator, self).__init__()

        # Hyperparameters
        self.channels_noise = channels_noise
        self.features = features
        self.channels_img = channels_img
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Blocks
        self.block1 = self._block(self.channels_noise, self.features * 4, 
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
            self.block3, # 32 x 14 x 14
            self.block4  # 1 x 28 x 28
        )
            
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

    def forward(self, x):
        return self.gen(x)

def invert(imgs, generator, device='mps', steps=100, predictor=None, encoder=None, 
            lr=0.1, weights=[1.0, 0.0, 0.0]):
    
    """
    Invert the input images into the generator's latent space using SGD.

    Inversion is the process of searching through the latent space to find where 
    real data is embedded after training a generative model. Performing stochastic 
    gradient descent on an initial vector converges towards a latent space 
    vector that can generate the original image.

    Regularizers, such as a supervised precitor or encoder, can ensure the 
    generated image has similar properties in other model feature spaces.
    However, for this example of a DCGAN using MNIST, inversion without 
    predictor and encoder loss is sufficient.

    It's assumed the generator, predictor, and encoder were all trained on
    the same data, including any post processing (e.g. MNIST scaled from -1 to 1
    instead of 0 to 1). If the predictor and encoder were not trained on the same 
    data scaling as the generator, retrain the models or change the source code.
    
    Parameters
    ----------
    imgs : np.array
        Input images with shape (number of images, height, width).
    generator : torch.nn.Module
        Generator model, typically from a generative adversarial network.
    device : string, default='mps'
        The device the generator is stored.
    steps : int, default=100
        Number of optimization steps.
    predictor : torch.nn.Module, default=None
        A model (e.g. CNN) that predicts some label for the image. This model
        ensures that the original image and generated image are similar in
        predictive feature space (e.g. the features from the last fully 
        connected layer).
    encoder : torch.nn.Module, default=None
        A model (e.g. encoder network for an autoencoder) that reduces the 
        dimensionality of an image. This model ensures that the original image
        and generated image are similar in encoded feature space.
    lr : float, default=0.1
        The optimizer's learning rate. A smaller rate generally requires more
        optimization steps to fully converge.
    weights : list of floats, default=[1.0, 0.0, 0.0]
        The weights applied when summing three different losses for the total 
        loss, respectively:
        - reconstruction loss (mean squared error between original image and 
        generated image in pixel space), 
        - the predictor loss (mean squared error between original image and 
        generated image in predictive feature space), and
        - the encoder loss (mean squared error between original image and 
        generated image in encoded feature space).

    Returns
    -------
    z : np.array
        The latent space vectors of the inverted images.
    x_gens : np.array
        If inverting only one image, the generated images at each optimization 
        step are returned. If inverting more, the generated images at the final
        optimization step are returned.
    loss : np.array
        The losses at each optimization step.
    """

    # Reshape image and unpack loss weights
    n_imgs = imgs.shape[0]
    img_size = imgs.shape[1]
    imgs_reshape = imgs.reshape(n_imgs, 1, img_size, img_size)
    loss_rec_weight, loss_pred_weight, loss_enc_weight = weights

    # Prepare image and latent space vector
    channels_noise = generator.channels_noise
    x = torch.Tensor(imgs_reshape).to(device)
    x.requires_grad = False
    z = torch.ones((n_imgs, channels_noise, 1, 1)).to(device)
    z.requires_grad = True

    # Prepare optimizer
    optimizer = torch.optim.Adam([z], lr=lr)
    
    # Invert image to latent space
    x_gens = []
    loss = []
    for step in tqdm(range(steps)):
        loss_total = 0.0
        loss_step = []

        # Reconstruction loss
        x_gen = generator(z)
        if n_imgs == 1:
            x_gens.append(x_gen.cpu().detach().numpy()[0,0])
        loss_rec = torch.mean((x - x_gen) ** 2)
        loss_rec_weighted = loss_rec * loss_rec_weight
        loss_step.append(loss_rec_weighted.cpu().detach().numpy())
        loss_total += loss_rec_weighted
        
        # Predictor loss
        if loss_pred_weight:
            x_pred = predictor(x)
            x_gen_pred = predictor(x_gen)
            loss_pred = torch.mean((x_pred - x_gen_pred) ** 2)
            loss_pred_weighted = loss_pred * loss_pred_weight
            loss_step.append(loss_pred_weighted.detach().numpy())
            loss_total += loss_pred_weighted

        # Encoder loss
        if loss_enc_weight:
            x_enc = encoder(x)
            x_gen_enc = encoder(x_gen)
            loss_enc = torch.mean((x_enc - x_gen_enc) ** 2)
            loss_enc_weighted = loss_enc * loss_enc_weight
            loss_step.append(loss_enc_weighted.detach().numpy())
            loss_total += loss_enc_weighted
            
        loss.append(loss_step)

        # Backprop
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    # Prepare returns
    z = z.cpu().detach().numpy()
    if n_imgs == 1:
        x_gens = np.array(x_gens)
    else:
        x_gens = x_gen.cpu().detach().numpy()
        x_gens = x_gens.reshape(n_imgs, img_size, img_size)
    loss = np.array(loss)
    
    return z, x_gens, loss