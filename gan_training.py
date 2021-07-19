import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from gan_loss import *
from visualize import visualize_tensor_images


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    gen,
    disc,
    gen_opt,
    disc_opt,
    dataloader,
    criterion=nn.BCEWithLogitsLoss(),
    display_step=50,
    z_dim = 64,
    n_epochs=50,
    n_batch=128,
    lr=0.0001,
    device=DEVICE):
    
    cur_step = 0
    mean_gen_loss = 0
    mean_disc_loss = 0
    gen_loss = False

    for epoch in range(n_epochs):

        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)
            disc_opt.zero_grad()
            disc_loss_val = disc_loss(
                gen,
                disc,
                criterion,
                real,
                cur_batch_size,
                z_dim,
                device=DEVICE)
            disc_loss_val.backward(retain_graph=True)
            disc_opt.step()
            gen_opt.zero_grad()
            gen_loss_val = gen_loss(
                gen,
                disc,
                criterion,
                cur_batch_size,
                z_dim,
                device=DEVICE)
            gen_loss_val.backward()
            gen_opt.step()
            mean_gen_loss += gen_loss_val.item() / display_step
            mean_disc_loss += disc_loss_val.item() / display_step
            
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    "Epoch: {}/{}".format(epoch, n_epochs),
                    "Gen_loss: {:.4f}".format(mean_gen_loss),
                    "Disc_loss: {:.4f}".format(mean_disc_loss))
                fake_noise = torch.randn(cur_batch_size, z_dim).to(device)
                fake_images = gen(fake_noise)
                visualize_tensor_images(fake_images)
                visualize_tensor_images(real)
                mean_gen_loss = 0
                mean_disc_loss = 0
            
            cur_step += 1
                

                
