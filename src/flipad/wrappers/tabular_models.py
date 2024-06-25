import logging
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# sys.path.append("src")
from flipad.networks.klwgan import KLWGANGenerator

EPS = 0.0001

logger = logging.getLogger(__name__)


class KLWGANWrapper:
    name = "klwgan"
    seed_dim = 10

    def __init__(self, output_dim, checkpoint_path: Path):
        self.nz = 10
        self.G = KLWGANGenerator(output_dim).cuda()
        try:
            logger.info(f"Loading checkpoint {checkpoint_path}.")
        except NameError:
            print("No logger found!")
            print(f"Loading checkpoint {checkpoint_path}.")
            pass
        self.G.load_state_dict(torch.load(checkpoint_path))
        self.checkpoint_path = checkpoint_path
        self.index_last_activation = (
            3  # The 3th layer is the last LeakyReLU, 4th is final Linear Layer.
        )

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
            z = torch.randn([batch_size, self.nz]).cuda()
            # iterate the random noise through all layers until the index_last_activation-th layer
            for name_layer, layer in self.G.named_modules():
                if name_layer not in [
                    "net",
                    "",
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
        return target  # No final activation

    def image_to_target(self, images: torch.Tensor) -> torch.Tensor:
        return images  # No final activation

    def get_last_linearity(self):
        """
        :return: nn.Module of last linear layer, type of convolution: "conv" or "tconv" or "fc"
        """
        return self.G.net[self.index_last_activation + 1], "fc"

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
