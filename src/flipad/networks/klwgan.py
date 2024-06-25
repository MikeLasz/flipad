from torch import nn

sn = nn.utils.spectral_norm


class KLWGANGenerator(nn.Module):
    def __init__(self, input_dim):
        super(KLWGANGenerator, self).__init__()
        self.noise_dim = 10
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim, 300),
            nn.LeakyReLU(),
            sn(nn.Linear(300, 300)),
            nn.LeakyReLU(),
            nn.Linear(300, input_dim),
        )

    def forward(self, z):
        return self.net(z)


class KLWGANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(KLWGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            sn(nn.Linear(input_dim, 300)),
            nn.LeakyReLU(),
            sn(nn.Linear(300, 300)),
            nn.LeakyReLU(),
            sn(nn.Linear(300, 1)),
        )

    def forward(self, x):
        return self.net.forward(x)
