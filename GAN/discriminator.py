import torch
import torch.nn as nn


def disc_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )


class Discriminator(nn.Module):

    def __init__(self,in_dim=784, hid_dim=128):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            disc_block(in_dim,hid_dim*4),
            disc_block(hid_dim*4, hid_dim * 2),
            disc_block(hid_dim * 2, hid_dim),
            nn.Linear(hid_dim, 1),

        )

    def forward(self, image):
        return self.disc(image)
