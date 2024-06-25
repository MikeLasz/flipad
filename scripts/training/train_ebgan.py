import argparse
import os
import pprint
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from flipad.networks.ebgan import EBGANDiscriminator, EBGANGenerator
from flipad.utils import weights_init
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
    "--outf",
    default="trained_models/ebgan",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--clip_value",
    type=float,
    default=0.01,
    help="lower and upper clip value for disc. weights",
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

# Reconstruction loss of AE
pixelwise_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
netG = EBGANGenerator(opt.nz, nc, opt.imageSize, use_conv=False)
netD = EBGANDiscriminator(nc, opt.imageSize, use_ae=True)

if cuda:
    netG.cuda()
    netD.cuda()
    pixelwise_loss.cuda()

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings**2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt


# ----------
#  Training
# ----------

# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, opt.batchSize / 64.0)

G_losses = []
D_losses = []
for epoch in range(opt.niter):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        if opt.dataset == "celeba":
            if netG.use_conv:
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz))))
            else:
                z = Variable(
                    Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz, 1, 1)))
                )
        else:
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.nz, 1, 1))))

        # Generate a batch of images
        gen_imgs = netG(z)
        recon_imgs, img_embeddings = netD(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = pixelwise_loss(
            recon_imgs, gen_imgs.detach()
        ) + lambda_pt * pullaway_loss(img_embeddings)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_recon, _ = netD(real_imgs)
        fake_recon, _ = netD(gen_imgs.detach())

        d_loss_real = pixelwise_loss(real_recon, real_imgs)
        d_loss_fake = pixelwise_loss(fake_recon, gen_imgs.detach())

        d_loss = d_loss_real
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += margin - d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # --------------
        # Log Progress
        # --------------

        # Save Losses for plotting later
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.niter, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(
                gen_imgs.data[:25],
                "%s/fake_samples_epoch_%03d.png" % (img_dir, epoch),
                nrow=5,
                normalize=True,
            )

    if (epoch + 1) % 5 == 0:
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
