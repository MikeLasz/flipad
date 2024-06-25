from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from sad.base.base_net import BaseNet

class CIFAR10_LeNet(BaseNet):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        #x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class CIFAR10_BigLeNet(BaseNet):
    def __init__(self, rep_dim=128, input_channels=3):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(input_channels, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 64, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv5 = None
        # self.fc1 = nn.Linear(64 * 16 * 16, self.rep_dim, bias=False) # for 256x256 input
        self.fc1 = nn.Linear(64 * 8 * 8, self.rep_dim, bias=False)  # for 128x128 input

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        if self.conv5 is not None:
            x = self.conv5(x)
            x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x

class CIFAR10_LeNet_Decoder(BaseNet):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class CIFAR10_LeNet_Autoencoder(BaseNet):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LeNet_128x128(BaseNet):
    def __init__(self, rep_dim=128, input_channels=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(input_channels, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 16, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(16, 8, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 8 * 8, self.rep_dim, bias=False)  # For inputs batch_size x 128 x 128 x 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class LeNet_64Channels(BaseNet):
    def __init__(self, rep_dim=128, input_spatial_size=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        assert input_spatial_size == 32 or input_spatial_size == 64
        self.input_spatial_size = input_spatial_size
        if self.input_spatial_size==64:
            # additional convolution that maps 64x64x64 to 64x32x32
            self.conv0 = nn.Conv2d(64, 64, 4, stride=2, padding=1, bias=False)

        self.conv1 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(256, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        # x = x.view(-1, 3, 32, 32)
        if self.input_spatial_size==64:
            x = F.leaky_relu(self.conv0(x))
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class LeNet_64Channels_Decoder(BaseNet):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 512, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d4 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(256, 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d6 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain("leaky_relu"))
        # check all channels....
        self.bn2d7 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(128, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain("leaky_relu"))
        self.bn2d8 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(128, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv6(x)
        x = torch.sigmoid(x)
        return x


class LeNet_64Channels_Autoencoder(BaseNet):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = LeNet_128x128(rep_dim=rep_dim)
        self.decoder = LeNet_64Channels_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
