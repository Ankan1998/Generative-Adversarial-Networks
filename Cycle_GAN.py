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


class discriminator(nn.Module):

    def __init__(self,):

        super().__init__()
        self.conv2d_1 = nn.Conv2d(3,64)


