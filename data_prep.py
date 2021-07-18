from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

def data_loader(path,batch_size):
    return DataLoader(
        MNIST(path,
              download=True,
              transform = transforms.Compose(
                  [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
              ),
        ),
        batch_size = batch_size,
        shuffle = True
    )

if __name__ == "__main__":
    pass