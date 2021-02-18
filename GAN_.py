import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt



writer_real =SummaryWriter("runs/mnist_gan/real")
writer_fake =SummaryWriter("runs/mnist_gan/fake")
step = 0

class Discriminator(nn.Module):

    def __init__(self,image_dim):

        super().__init__()
        self.linear1 = nn.Linear(image_dim,128)
        self.leakyrelu = nn.LeakyReLU(0.01)
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
        self.leakyrelu = nn.LeakyReLU(0.01)
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
batch_size = 32
num_epochs = 100

z_dim =100

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim,image_dim).to(device)

latent_noise = torch.randn(batch_size,z_dim).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms,download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle =True)


disc_opt =  torch.optim.Adam(disc.parameters(),lr=lr)
gen_opt =  torch.optim.Adam(gen.parameters(),lr=lr)

criterion = nn.BCELoss()

for epoch in tqdm(range(num_epochs)):
    for idx, (real,_) in enumerate(loader):
      real = real.view(-1,784).to(device)
      batch_size = real.shape[0]

      # Discriminator

      noise = torch.randn(batch_size,z_dim).to(device)
      fake = gen(noise)

      disc_real = disc(real).view(-1)
      loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
      disc_fake = disc(fake.detach()).view(-1)
      loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

      loss_disc = (loss_disc_real + loss_disc_fake)/2

      disc.zero_grad()
      loss_disc.backward()
      disc_opt.step()


      # Generator
      out = disc(fake).view(-1)
      loss_gen = criterion(out,torch.ones_like(out))

      gen.zero_grad()
      loss_gen.backward()
      gen_opt.step()

      if idx == 0:
          print(
              f"Epoch [{epoch}/{num_epochs}] Batch {idx}/{len(loader)} \
                            Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
          )

          with torch.no_grad():
              fake = gen(latent_noise).reshape(-1, 1, 28, 28)
              data = real.reshape(-1, 1, 28, 28)


              img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
              img_grid_real = torchvision.utils.make_grid(data, normalize=True)
              # plt.imshow(img_grid_fake.permute(1,2,0))
              # plt.show()
              writer_fake.add_image(
                  "Mnist Fake Images", img_grid_fake, global_step=step
              )
              writer_real.add_image(
                  "Mnist Real Images", img_grid_real, global_step=step
              )
              step += 1



























