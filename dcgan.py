import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)

class Discriminator_SN(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator_SN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator_SN_wide3(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator_SN_wide3, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, (4,3), (2,3), (1,2), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, (5,4), (1,4), (2,2), bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, (3,4), (1,4), (1,2), bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def training_loop(num_epochs, dataloader, save_discr, wide_images, netG, netD, device, criterion, nz, smooth, optimizerG, optimizerD, fixed_noise, out, model_save_path, model_save_freq):
    netG.train()
    netD.train()
    img_list = []
    img_list_only = []
    G_losses = []
    D_losses = []
    iters = 0
    
    if smooth == True:
        real_label = 0.9
        fake_label = 0.1
    else:
        real_label = 1.
        fake_label = 0.

    format_epoch = len(str(num_epochs))
    format_iterations = len(str(len(dataloader)))
    os.makedirs(out, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch:0{format_epoch}d}/{num_epochs}][{i:0{format_iterations}d}/{len(dataloader)}]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % model_save_freq == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                if wide_images == True:
                    img_list.append(vutils.make_grid(fake[:8], padding=2, normalize=True, nrow = 1))
                    save_image(vutils.make_grid(fake[:8], padding=2, normalize=True, nrow = 1),
                                    out+'/it_{:05d}_epoch_{:04d}_grid_img.png'.format(iters, epoch+1))
                else:
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    save_image(vutils.make_grid(fake, padding=2, normalize=True),
                                out+'/it_{:05d}_epoch_{:04d}_grid_img.png'.format(iters, epoch+1))
                img_list_only.append(fake)

            if (iters % model_save_freq == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                torch.save(netG.state_dict(), f'{model_save_path}/Epoch_{epoch:03d}_Iter_{iters:06d}.zip')
                if save_discr == True:
                    torch.save(netD.state_dict(), f'{model_save_path}/Epoch_{epoch:03d}_Iter_{iters:06d}_discr.zip')
            iters += 1

        if max(D_losses[-10:]) < 0.0005:
            print('End training due to failed GAN-game')
            break

    return G_losses, D_losses, img_list, img_list_only