import torch
import torch.nn as nn


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class EBGANGenerator(nn.Module):
    def __init__(self, nz, nc, img_size=64, use_conv=False):
        super(EBGANGenerator, self).__init__()

        self.img_size = img_size
        self.init_size = self.img_size // 4
        self.use_conv = use_conv
        ngf = 64
        if use_conv:
            self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))
            self.conv_blocks = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf * 2, 0.8),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf, 0.8),
                nn.ReLU(inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                nn.Tanh(),
            )
        else:
            # use transposed convolutions
            self.conv_blocks = nn.Sequential(
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
                nn.Tanh(),
            )

    def forward(self, noise):
        if len(noise.shape) == 2:
            noise = noise[:, :, None, None]
        if self.use_conv:
            out = self.l1(noise)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
        else:
            if len(noise.shape) == 2:
                noise = noise.unsqueeze(2).unsqueeze(3)
            img = self.conv_blocks(noise)
        return img


class EBGANDiscriminator(nn.Module):
    def __init__(self, nc, img_size=64, use_ae=True):
        super(EBGANDiscriminator, self).__init__()

        self.img_size = img_size
        self.use_ae = use_ae
        if self.use_ae:
            self._enc = nn.Sequential(
                # 3 x 64 x 64
                nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # ndf x 32 x 32
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # (ndf*2) x 16 x 16
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                # (ndf*4) x 8 x 8
            )
            self._dec = nn.Sequential(
                # (ndf*4) x 8 x 8
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                # (ndf*2) x 16 x 16
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # (ndf) x 32 x 32
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                # (3) x 64 x 64
            )
        else:
            # Upsampling
            self.down = nn.Sequential(nn.Conv2d(nc, 64, 3, 2, 1), nn.ReLU())
            # Fully-connected layers
            self.down_size = self.img_size // 2
            down_dim = 64 * (self.img_size // 2) ** 2

            self.embedding = nn.Linear(down_dim, 32)

            self.fc = nn.Sequential(
                nn.BatchNorm1d(32, 0.8),
                nn.ReLU(inplace=True),
                nn.Linear(32, down_dim),
                nn.BatchNorm1d(down_dim),
                nn.ReLU(inplace=True),
            )
            # Upsampling
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2), nn.Conv2d(64, nc, 3, 1, 1)
            )

    def forward(self, img):
        if self.use_ae:
            output = self._enc(img)
            embedding = output.view(-1, 64 * 4, 8, 8)
            output = self._dec(embedding)
            return output, embedding.flatten(1)
        else:
            out = self.down(img)
            embedding = self.embedding(out.view(out.size(0), -1))
            out = self.fc(embedding)
            out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
            return out, embedding


class gen(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(gen, self).__init__()
        self._gen = nn.Sequential(
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
            nn.Tanh(),
        )

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(2).unsqueeze(3)
        output = self._gen(input)
        return output


class enc(nn.Module):
    def __init__(self, ndf, nc):
        super(enc, self).__init__()
        self._enc = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # (ndf*4) x 8 x 8
        )

    def forward(self, input):
        output = self._enc(input)
        return output.view(-1, 1)


class dec(nn.Module):
    def __init__(self, ndf):
        super(dec, self).__init__()
        self._dec = nn.Sequential(
            # (ndf*4) x 8 x 8
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # (ndf*2) x 16 x 16
            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # (ndf) x 32 x 32
            nn.ConvTranspose2d(ndf, 3, 4, 2, 1, bias=False),
            # (3) x 64 x 64
        )

    def forward(self, input):
        output = self._dec(input)
        return output


class autoenc(nn.Module):
    def __init__(self, ndf, nc):
        super(autoenc, self).__init__()
        self.ndf = ndf
        self._enc = enc(ndf, nc)
        self._dec = dec(ndf)

    def forward(self, input):
        output = self._enc(input)
        embedding = output.view(-1, self.ndf * 4, 8, 8)
        output = self._dec(embedding)
        return output, embedding.flatten(1)
