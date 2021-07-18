import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_dim=784, hid_dim=128):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.disc_block(in_dim, hid_dim * 4),
            self.disc_block(hid_dim * 4, hid_dim * 2),
            self.disc_block(hid_dim * 2, hid_dim),
            nn.Linear(hid_dim, 1),

        )

    def disc_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        return self.disc(image)


if __name__ == "__main__":
    pass
