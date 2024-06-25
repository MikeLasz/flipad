from __future__ import print_function

import argparse
import os
import pprint
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from matplotlib import pyplot as plt

from flipad.networks.dcgan import DCGANDiscriminator, DCGANGenerator
from flipad.data_utils import get_data_loader
from flipad.utils import weights_init

parser = argparse.ArgumentParser()
parser.add_argument("data_root")
parser.add_argument(
    "--dataset",
    required=True,
    help="cifar10 | lsun | mnist | imagenet | folder | lfw | fake | celeba",
)
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=4
)
parser.add_argument("--batchSize", type=int, default=128, help="input batch size")
parser.add_argument(
    "--imageSize",
    type=int,
    default=64,
    help="the height / width of the input image to network",
)
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument(
    "--niter", type=int, default=25, help="number of epochs to train for"
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
)
parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
)
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument(
    "--dry-run", action="store_true", help="check a single training cycle works"
)
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--netG", default="", help="path to netG (to continue training)")
parser.add_argument("--netD", default="", help="path to netD (to continue training)")
parser.add_argument(
    "--outf",
    default="trained_models/dcgan",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--manualSeed", type=int, help="manual seed")

opt = parser.parse_args()
print(opt)

# directories
model_nr = 1  # initialize model_nr
output_dir = os.path.join(
    opt.outf, opt.dataset, f"nz={opt.nz}_niter={opt.niter}_model={model_nr}"
)
while os.path.exists(output_dir):
    # iterate through model_nr if model already exists
    model_nr += 1
    output_dir = os.path.join(
        opt.outf, opt.dataset, f"nz={opt.nz}_niter={opt.niter}_model={model_nr}"
    )
img_dir = os.path.join(output_dir, "images")
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(output_dir)
os.makedirs(img_dir)
os.makedirs(checkpoint_dir)

# seeding
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# save config
with open(os.path.join(output_dir, "opt.txt"), "w") as file:
    file.write(pprint.pformat(vars(opt)))

# cuda
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

# setup data
dataloader = get_data_loader(
    data=opt.dataset,
    root=opt.data_root,
    batch_size=opt.batchSize,
    num_workers=opt.workers,
)
nc = 1 if opt.dataset == "mnist" else 3
fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)
# soft labels
real_label = 0.9
fake_label = 0.1

# setup network
netG = DCGANGenerator(opt.nz, opt.ngf, nc, opt.ngpu).to(device)
netG.apply(weights_init)
if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = DCGANDiscriminator(opt.nz, opt.ndf, nc, opt.ngpu).to(device)
netD.apply(weights_init)
if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# setup optimizer
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# train
if opt.dry_run:
    opt.niter = 1

G_losses = []
D_losses = []
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full(
            (batch_size,), real_label, dtype=real_cpu.dtype, device=device
        )

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        print(
            "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (
                epoch,
                opt.niter,
                i,
                len(dataloader),
                errD.item(),
                errG.item(),
                D_x,
                D_G_z1,
                D_G_z2,
            )
        )
        if i % 100 == 0:
            vutils.save_image(real_cpu, "%s/real_samples.png" % img_dir, normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(
                fake.detach(),
                "%s/fake_samples_epoch_%03d.png" % (img_dir, epoch),
                normalize=True,
            )

        if opt.dry_run:
            break

    # do checkpointing
    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % (checkpoint_dir, epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % (checkpoint_dir, epoch))

# plot losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "losses.png"))
