import argparse
import os
import pprint
import random

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from flipad.networks.wgan import (
    WGANDiscriminator,
    WGANGenerator,
    compute_gradient_penalty,
)
from flipad.data_utils import get_data_loader

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
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument(
    "--outf",
    default="trained_models/wgangp",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval betwen image samples"
)
opt = parser.parse_args()
print(opt)

img_shape = (
    (1, opt.imageSize, opt.imageSize)
    if opt.dataset == "mnist"
    else (3, opt.imageSize, opt.imageSize)
)

cuda = True if torch.cuda.is_available() else False
if cuda:
    device = "cuda"
else:
    device = "cpu"
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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

# save config
with open(os.path.join(output_dir, "opt.txt"), "w") as file:
    file.write(pprint.pformat(vars(opt)))

# seeding
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# setup data
dataloader = get_data_loader(
    data=opt.dataset,
    root=opt.data_root,
    batch_size=opt.batchSize,
    num_workers=opt.workers,
)
nc = 1 if opt.dataset == "mnist" else 3
fixed_noise = torch.randn(opt.batchSize, opt.nz, 1, 1, device=device)

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
netG = WGANGenerator(opt.nz, opt.ngf, nc)
netD = WGANDiscriminator(opt.ndf, nc)

if cuda:
    netG.cuda()
    netD.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# ----------
#  Training
# ----------
G_losses = []
D_losses = []
batches_done = 0
for epoch in range(opt.niter):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input

        z = torch.randn(
            real_imgs.shape[0], opt.nz, 1, 1, device=device
        )  # Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz))))
        # Variable is deprecated: Its use was to allow for automatic differnatioation...

        # Generate a batch of images
        fake_imgs = netG(z)

        # Real images
        real_validity = netD(real_imgs)
        # Fake images
        fake_validity = netD(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            netD, real_imgs.data, fake_imgs.data
        )
        # Adversarial loss
        d_loss = (
            -torch.mean(real_validity)
            + torch.mean(fake_validity)
            + lambda_gp * gradient_penalty
        )

        d_loss.backward()
        optimizer_D.step()
        D_losses.append(d_loss.item())

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = netG(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = netD(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()
            # update G_losses with the same loss opt.n_critic times to ensure that D_losses and G_losses have the same size
            for n in range(opt.n_critic):
                G_losses.append(g_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.niter, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                save_image(
                    fake_imgs.data[:25],
                    "%s/fake_samples_epoch_%03d.png" % (img_dir, epoch),
                    nrow=5,
                    normalize=True,
                )

            batches_done += opt.n_critic

    if (epoch + 1) % 10 == 0:
        # do checkpointing
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
