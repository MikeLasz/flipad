import torch
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class LSGANGenerator(nn.Module):
    def __init__(self, nz, nc, img_size=64, size="big"):
        super(LSGANGenerator, self).__init__()
        self.img_size = img_size
        if size == "small":
            self.init_size = self.img_size // 4
        else:
            self.init_size = self.img_size // 8
        self.l1 = nn.Sequential(nn.Linear(nz, 128 * self.init_size**2))

        if size == "small":
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, nc, 3, stride=1, padding=1),
                nn.Tanh(),
            )
        else:
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, nc, 3, stride=1, padding=1),
                nn.Tanh(),
            )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class LSGANDiscriminator(nn.Module):
    def __init__(self, nc, img_size=64, size="small"):
        super(LSGANDiscriminator, self).__init__()

        self.img_size = img_size

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        if size == "small":
            self.model = nn.Sequential(
                *discriminator_block(nc, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
            )
        else:
            self.model = nn.Sequential(
                *discriminator_block(nc, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128),
                *discriminator_block(128, 128),
            )

        # The height and width of downsampled image
        if size == "small":
            ds_size = self.img_size // 2**4
        else:
            ds_size = self.img_size // 2**5
        self.adv_layer = nn.Linear(128 * ds_size**2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DiscriminatorModel(nn.Module):
    def __init__(self, n_c, n_fmps):
        super(DiscriminatorModel, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_c, n_fmps, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_fmps),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_fmps, n_fmps * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_fmps * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_fmps * 2, n_fmps * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_fmps * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_fmps * 4, n_fmps * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(n_fmps * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_fmps * 8, 1, 4, stride=1, padding=0),
        )
        self.linear = nn.Linear(1, 1)
        self.net.apply(weights_init)
        self.linear.apply(weights_init)

    def forward(self, x):
        x = self.net(x)
        x = self.linear(x)
        return x.view(-1, 1, 1, 1)


class GeneratorModel(nn.Module):
    def __init__(self, n_z, n_fmps, n_c):
        super(GeneratorModel, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_fmps * 8, 4, 1, 0),
            nn.BatchNorm2d(n_fmps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fmps * 8, n_fmps * 4, 4, 2, 1),
            nn.BatchNorm2d(n_fmps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fmps * 4, n_fmps * 2, 4, 2, 1),
            nn.BatchNorm2d(n_fmps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fmps * 2, n_fmps * 1, 4, 2, 1),
            nn.BatchNorm2d(n_fmps * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_fmps, n_c, 4, 2, 1),
            nn.Tanh(),
        )
        self.net.apply(weights_init)

    def forward(self, x):
        x = self.net(x)
        return x
