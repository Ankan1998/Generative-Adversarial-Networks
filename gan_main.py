import torch
import torch.nn as nn
from GAN.discriminator import Discriminator
from GAN.generator import Generator
from data_prep import data_loader
import torch.optim as optim
from gan_training import train_gan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def training_gan(
    path = "dataset",
    n_epochs = 200,
    z_dim = 64,
    display_step = 500,
    batch_size = 128,
    lr = 0.00001,
    device=DEVICE):

    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)
    optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
    optimizer_disc = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataloader = data_loader(path,batch_size)

    train_gan(
        gen,
        disc,
        optimizer_gen,
        optimizer_disc,
        dataloader,
        criterion,
        display_step,
        z_dim,
        n_epochs,
        batch_size,
        lr,
        device)


if __name__ == '__main__':
    training_gan()