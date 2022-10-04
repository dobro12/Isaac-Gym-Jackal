from encoder import Encoder
from decoder import Decoder

from torchvision.utils import save_image
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets
import numpy as np
import argparse
import wandb
import torch
import cv2
import os

def loss_fn(recon_x, x, mu, log_var):
    bce_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
    total_loss = bce_loss + kl_loss
    return total_loss, bce_loss, kl_loss

def save(encoder, decoder, suffix=''):
    if not os.path.exists("model"):
        os.mkdir("model")
    torch.save(
        {'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),}
        ,f"model/checkpoint{suffix}")

def load(encoder, decoder, suffix=''):
    checkpoint = torch.load(f"model/checkpoint{suffix}")
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

def train():
    # parameter
    batch_size = 256
    learning_rate = 1e-3
    epochs = 100
    save_interval = 1

    # for wandb
    wandb.init(project='[Isaac-Gym-Jackal] VAE')

    # define torch device
    device = torch.device("cuda:0")

    # Load Data
    dataset = datasets.ImageFolder(root='img/', 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define VAE
    encoder = Encoder().to(device=device)
    decoder = Decoder().to(device=device)

    # define optimizaer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for epoch in range(epochs):
        total_losses = []
        bce_losses = []
        kl_losses = []
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device=device)
            mean, log_var, std = encoder(images)
            esp = torch.randn(*mean.size(), device=device)
            latents = mean + std * esp
            recon_images = decoder(latents)

            total_loss, bce_loss, kl_loss = loss_fn(recon_images, images, mean, log_var)
            total_losses.append(total_loss.item())
            bce_losses.append(bce_loss.item())
            kl_losses.append(kl_loss.item())
  
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        log = {"total_loss":np.mean(total_losses), "bce_loss":np.mean(bce_losses), "kl_loss":np.mean(kl_losses)}
        wandb.log(log)
        print(log)

        if (epoch + 1)%save_interval == 0:
            # save(encoder, decoder, f"{(epoch+1)}")
            save(encoder, decoder)

def test(resume):
    #define parameter
    batch_size = 4

    # define torch device
    device = torch.device("cuda:0")

    # Load Data
    dataset = datasets.ImageFolder(root='img/', 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define VAE
    encoder = Encoder().to(device=device)
    decoder = Decoder().to(device=device)
    load(encoder, decoder, resume)

    images, _ = next(iter(dataloader))
    images = images.to(device=device)
    mean, log_var, std = encoder(images)
    recon_images = decoder(mean)
    concat_imgs = torch.cat([images, recon_images], dim=2).cpu()
    concat_imgs = torch.cat([concat_imgs[i] for i in range(batch_size)], dim=-1)
    save_image(concat_imgs, 'sample_image.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE scripts.')
    parser.add_argument('--test', action='store_true',
                        help='For test.')
    parser.add_argument('--resume', type=int, default=0,
                        help='type # of checkpoint.')
    args = parser.parse_args()
    if args.test:
        test(args.resume)
    else:
        train()