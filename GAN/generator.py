import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, latent_noise_dim=10, in_dim=784, hid_dim=128):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.gen_block(latent_noise_dim,hid_dim),
            self.gen_block(hid_dim, hid_dim * 2),
            self.gen_block(hid_dim * 2, hid_dim * 4),
            self.gen_block(hid_dim * 4, hid_dim * 8),
            nn.Linear(hid_dim * 8, in_dim),
            nn.Sigmoid()

        )

    def gen_block(self,input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, noise):
        return self.gen(noise)

if __name__ == "__main__":
    pass