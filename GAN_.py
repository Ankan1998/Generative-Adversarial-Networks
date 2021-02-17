import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Discriminator(nn.Module):

    def __init__(self,image_dim):

        super().__init__()
        self.linear1 = nn.Linear(image_dim,128)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(128,1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):

        x = self.linear1(x)
        x = self.leakyrelu(x)
        x = self.linear2(x)
        x = self.Sigmoid(x)

        return x



class Generator(nn.Module):

    def __init__(self, latent_noise_dim,img_dim):

        super().__init__()

        self.linear1 = nn.Linear(latent_noise_dim,128)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(128,img_dim)
        self.tanh = nn.Tanh()

    def forward(self,x):

        x = self.linear1(x)
        x = self.leakyrelu(x)
        x = self.linear2(x)
        x = self.tanh(x)

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
lr=3e-4
image_dim = 28 * 28 * 1
batch_size=32
num_epochs = 10

latent_noise_dim = 28 * 28 * 1

disc = Discriminator(image_dim)
gen = Generator(latent_noise_dim,image_dim)

#latent_noise = torch.randn(batch_size,latent_noise_dim).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms,download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle =True)


disc_opt =  torch.optim.Adam(disc.parameters(),lr=lr)
gen_opt =  torch.optim.Adam(gen.parameters(),lr=lr)

criterion = nn.BCELoss()
















