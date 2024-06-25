import logging
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# sys.path.append("src")
from flipad.networks.medigan_dcgan import Generator

EPS = 0.0001

logger = logging.getLogger(__name__)


class MEDIDCGANWrapper:
    name = "medigan_dcgan"
    seed_dim = 100

    def __init__(self, checkpoint_path: Path):
        self.nz = 100
        self.G = Generator(
            nz=self.nz,
            ngf=64,
            nc=1,
            ngpu=1,
            image_size=128,
            conditional=False,
            leakiness=0.1,
        ).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass
        self.G.load_state_dict(torch.load(checkpoint_path))
        """
        I tested whether defining the model + loading state dict is equivalent to loading the model directly.
        There is a little difference of order e-5 per pixel but the resulting images look the same: 
        using_model.png vs using_selfg.png 
        """
        self.checkpoint_path = checkpoint_path
        self.index_last_activation = 14

    def sample_images(self, num_samples: int, batch_size: int) -> torch.Tensor:
        images = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling images"
        ):
            z = torch.randn([batch_size, self.nz, 1, 1]).cuda()
            img = self.G(z)
            images.append(img.detach().cpu())
        return torch.cat(images)[:num_samples]

    def sample_activations(self, num_samples: int, batch_size: int = 1) -> torch.Tensor:
        activations = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling activations"
        ):
            counter_layer = 0
            z = torch.randn([batch_size, self.nz, 1, 1]).cuda()
            # iterate the random noise through all layers until the index_last_activation-th layer
            for layer in self.G.main:
                z = layer(z)
                counter_layer += 1
                if counter_layer > self.index_last_activation:
                    break
            activations.append(z.detach().cpu())
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.get_last_linearity()[0](activations)

    def target_to_image(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tanh(target)

    def image_to_target(self, images: torch.Tensor) -> torch.Tensor:
        return torch.atanh(torch.clip(images, min=-1 + EPS, max=1 - EPS))

    def get_last_linearity(self):
        """
        :return: nn.Module of last linear layer, type of convolution: "conv" or "tconv"
        """
        return self.G.main[self.index_last_activation + 1], "tconv"

    @staticmethod
    def float2pix(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * 127.5 + 128).clamp(0, 255)
