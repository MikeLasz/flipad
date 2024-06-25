import logging
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# sys.path.append("src")
from flipad.networks.dcgan import DCGANGenerator
from flipad.networks.ebgan import EBGANGenerator
from flipad.networks.lsgan import LSGANGenerator
from flipad.networks.wgan import WGANGenerator
from flipad.optimization import fista_reconstruct

EPS = 0.0001

logger = logging.getLogger(__name__)


class DCGANWrapper:
    name = "dcgan"
    seed_dim = 100

    def __init__(self, checkpoint_path: Path):
        self.nz = 100
        self.G = DCGANGenerator(self.nz, 64, 3, 1).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass
        self.G.load_state_dict(torch.load(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.index_last_activation = 11  # The 11th layer is the last ReLU, 12th is ConvTransposed2d, 13th is Tanh.

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
            for name_layer, layer in self.G.named_modules():
                if name_layer not in [
                    "",
                    "main",
                ]:  # the first named module is the full network, the second named module is the sequential([layer1, layer2, ...])
                    z = layer(z)
                    counter_layer += 1
                    if counter_layer > self.index_last_activation:
                        break
            activations.append(z.detach().cpu())
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.get_last_linearity[0](activations)

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


class WGANGPWrapper:
    name = "wgangp"
    seed_dim = 100

    def __init__(self, checkpoint_path: Path):
        self.nz = 100
        self.G = WGANGenerator(self.nz, 64, 3).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass

        # with open(checkpoint_path / CHECKPOINTS[dataset], "rb") as f:
        self.G.load_state_dict(torch.load(checkpoint_path))
        # self.G.synthesis.__class__.forward = patched_synthesis_forward

        self.checkpoint_path = checkpoint_path
        self.index_last_activation = 11  # The 11th layer is the last ReLU, 12th is ConvTransposed2d, 13th is Tanh.

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
            for name_layer, layer in self.G.named_modules():
                if name_layer not in [
                    "",
                    "main",
                ]:  # the first named module is the full network, the second named module is the sequential([layer1, layer2, ...])
                    z = layer(z)
                    counter_layer += 1
                    if counter_layer > self.index_last_activation:
                        break
            activations.append(z.detach().cpu())
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.get_last_linearity[0](activations)

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


class LSGANWrapper:
    name = "lsgan"
    seed_dim = 100

    def __init__(self, checkpoint_path: Path):
        self.nz = 100
        self.G = LSGANGenerator(self.nz, 3).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass

        # with open(checkpoint_path / CHECKPOINTS[dataset], "rb") as f:
        self.G.load_state_dict(torch.load(checkpoint_path))
        # self.G.synthesis.__class__.forward = patched_synthesis_forward

        self.checkpoint_path = checkpoint_path
        self.index_last_activation = len(self.G.conv_blocks) - 3

    def sample_images(self, num_samples: int, batch_size: int) -> torch.Tensor:
        images = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling images"
        ):
            z = torch.randn([batch_size, self.nz]).cuda()
            img = self.G(z)
            images.append(img.detach().cpu())
        return torch.cat(images)[:num_samples]

    def sample_activations(self, num_samples: int, batch_size: int = 1) -> torch.Tensor:
        activations = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling activations"
        ):
            counter_layer = 0
            z = torch.randn([batch_size, self.nz]).cuda()
            z = self.G.l1(z)
            z = z.view(z.shape[0], 128, self.G.init_size, self.G.init_size)
            for j in range(self.index_last_activation + 1):
                block = self.G.conv_blocks[j]
                z = block(z)
            activations.append(z.detach().cpu())
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.get_last_linearity[0](activations)

    def target_to_image(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tanh(target)

    def image_to_target(self, images: torch.Tensor) -> torch.Tensor:
        return torch.atanh(torch.clip(images, min=-1 + EPS, max=1 - EPS))

    def get_last_linearity(self):
        """
        :return: nn.Module of last linear layer, type of convolution: "conv" or "tconv"
        """
        return self.G.conv_blocks[self.index_last_activation + 1], "conv"

    @staticmethod
    def float2pix(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * 127.5 + 128).clamp(0, 255)


class EBGANWrapper:
    name = "ebgan"
    seed_dim = 100

    def __init__(self, checkpoint_path: Path):
        self.nz = 100
        self.G = EBGANGenerator(self.nz, 3, use_conv=False).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass

        self.G.load_state_dict(torch.load(checkpoint_path))

        self.checkpoint_path = checkpoint_path
        self.index_last_activation = len(self.G.conv_blocks) - 3

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
            z = torch.randn([batch_size, self.nz, 1, 1]).cuda()
            for j in range(self.index_last_activation + 1):
                block = self.G.conv_blocks[j]
                z = block(z)

            activations.append(z.detach().cpu())
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.get_last_linearity[0](activations)

    def target_to_image(self, target: torch.Tensor) -> torch.Tensor:
        return torch.tanh(target)

    def image_to_target(self, images: torch.Tensor) -> torch.Tensor:
        return torch.atanh(torch.clip(images, min=-1 + EPS, max=1 - EPS))

    def get_last_linearity(self):
        """
        :return: nn.Module of last linear layer, type of convolution: "conv" or "tconv"
        """
        conv_type = "conv" if self.G.use_conv else "tconv"
        return self.G.conv_blocks[self.index_last_activation + 1], conv_type

    @staticmethod
    def float2pix(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor * 127.5 + 128).clamp(0, 255)


if __name__ == "__main__":
    checkpoints = {
        "celeba": {
            "dcgan": "trained_models/dcgan/celeba/nz=100_niter=50_model=1/checkpoints/netG_epoch_49.pth",
            "wgangp": "trained_models/wgan_gp/celeba/nz=100_niter=200_model=1/checkpoints/netG_epoch_199.pth",
            "lsgan": "trained_models/lsgan/celeba/nz=100_niter=100_model=1/checkpoints/netG_epoch_99.pth",
            "ebgan": "trained_models/ebgan/celeba/nz=100_niter=100_model=1/checkpoints/netG_epoch_99.pth",
        },
        "lsun": {
            "dcgan": "trained_models/dcgan/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth",
            "wgangp": "trained_models/wgan_gp/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth",
            "lsgan": "trained_models/lsgan/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth",
            "ebgan": "trained_models/ebgan/lsun/nz=100_niter=10_model=1/checkpoints/netG_epoch_9.pth",
        },
    }
    for dataset in ["celeba", "lsun"]:
        # DCGAN
        wrapper = DCGANWrapper(checkpoint_path=Path(checkpoints[dataset]["dcgan"]))
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=16, batch_size=4)
        images = wrapper.float2pix(images) / 255
        save_image(images, f"debug/{dataset}_dcgan_images_original.png")

        torch.manual_seed(10)
        act = wrapper.sample_activations(num_samples=16, batch_size=4)

        output_with = wrapper.target(act.cuda())
        output_with = wrapper.float2pix(output_with) / 255
        save_image(output_with, f"debug/{dataset}_dcgan_images_from_act.png")

        # fista reconstructions
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=16, batch_size=4)
        recon_act = fista_reconstruct(images, wrapper, act.mean(dim=0))
        images_fista = wrapper.target(recon_act.cuda())
        images_fista = wrapper.float2pix(images_fista) / 255
        save_image(images_fista, f"debug/{dataset}_dcgan_images_fista.png")

        # WGANGP
        wrapper = WGANGPWrapper(checkpoint_path=Path(checkpoints[dataset]["wgangp"]))
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=16, batch_size=4)
        images = wrapper.float2pix(images) / 255
        save_image(images, f"debug/{dataset}_wgangp_images_original.png")

        torch.manual_seed(10)
        act = wrapper.sample_activations(num_samples=16, batch_size=4)
        output_with = wrapper.target(act.cuda())
        output_with = wrapper.float2pix(output_with) / 255
        save_image(output_with, f"debug/{dataset}_wgangp_images_from_act.png")

        # fista reconstructions
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=128, batch_size=16)
        recon_act = fista_reconstruct(images, wrapper, act.mean(dim=0))
        images_fista = wrapper.target(recon_act.cuda())
        images_fista = wrapper.float2pix(images_fista) / 255
        save_image(images_fista, f"debug/{dataset}_wgangp_images_fista.png")

        # LSGAN
        wrapper = LSGANWrapper(checkpoint_path=Path(checkpoints[dataset]["lsgan"]))
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=6, batch_size=4)
        images = wrapper.float2pix(images) / 255
        save_image(images, f"debug/{dataset}_lsgan_images_original.png")

        torch.manual_seed(10)
        act = wrapper.sample_activations(num_samples=16, batch_size=4)
        output_with = wrapper.target(act.cuda())
        output_with = wrapper.float2pix(output_with) / 255
        save_image(output_with, f"debug/{dataset}_lsgan_images_from_act.png")

        # fista reconstructions
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=128, batch_size=16)
        recon_act = fista_reconstruct(images, wrapper, act.mean(dim=0))
        images_fista = wrapper.target(recon_act.cuda())
        images_fista = wrapper.float2pix(images_fista) / 255
        save_image(images_fista, f"debug/{dataset}_lsgan_images_fista.png")

        # EBGAN
        wrapper = EBGANWrapper(checkpoint_path=Path(checkpoints[dataset]["ebgan"]))
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=16, batch_size=4)
        images = wrapper.float2pix(images) / 255
        save_image(images, f"debug/{dataset}_ebgan_images_original.png")

        torch.manual_seed(10)
        act = wrapper.sample_activations(num_samples=16, batch_size=4)
        output_with = wrapper.target(act.cuda())
        output_with = wrapper.float2pix(output_with) / 255
        save_image(output_with, f"debug/{dataset}_ebgan_images_from_act.png")

        # fista reconstructions
        torch.manual_seed(10)
        images = wrapper.sample_images(num_samples=128, batch_size=16)
        recon_act = fista_reconstruct(images, wrapper, act.mean(dim=0))
        images_fista = wrapper.target(recon_act.cuda())
        images_fista = wrapper.float2pix(images_fista) / 255
        save_image(images_fista, f"debug/{dataset}_ebgan_images_fista.png")
