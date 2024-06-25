import logging
import math
import pickle
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# sys.path.append("src/stylegan3")
print("test0")
from src.stylegan3.torch_utils import misc
import sys

sys.path.append("src/stylegan3")
# from stylegan3.torch_utils import misc
# from stylegan3.training.networks_stylegan2 import (FullyConnectedLayer, bias_act,
#                                         modulated_conv2d)

logger = logging.getLogger(__name__)


def patched_synthesis_forward(self, ws, **block_kwargs):
    block_ws = []
    with torch.autograd.profiler.record_function("split_ws"):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

    block_x = []
    block_img = []

    x = img = None
    for res, cur_ws in zip(self.block_resolutions, block_ws):
        block = getattr(self, f"b{res}")
        x, img = block(x, img, cur_ws, **block_kwargs)
        block_x.append(x)
        block_img.append(img)
    return img, block_x, block_img


class StyleGANWrapper:
    name = "stylegan2"
    seed_dim = 512

    def __init__(self, checkpoint_path: Path):
        logger.info(f"Loading checkpoint {checkpoint_path}.")
        with open(checkpoint_path, "rb") as f:
            self.G = pickle.load(f)["G_ema"].cuda()
        self.G.synthesis.__class__.forward = patched_synthesis_forward
        self.last_conv = torch.nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1, bias=False
        ).cuda()
        self.last_conv.weight = torch.nn.Parameter(self.G.synthesis.b256.torgb.weight)
        self.last_bias = self.G.synthesis.b256.torgb.bias.cuda()
        self.checkpoint_path = checkpoint_path

    def forward(self, input: torch.Tensor):
        return self.G(input, None)[0]

    def sample_images(self, num_samples: int, batch_size: int) -> torch.Tensor:
        images = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling images"
        ):
            z = torch.randn([batch_size, self.G.z_dim]).cuda()
            img, block_x, block_img = self.G(z, None)
            images.append(img.detach().cpu())
        return torch.cat(images)[:num_samples]

    def sample_activations(self, num_samples: int, batch_size: int = 1) -> torch.Tensor:
        activations = []
        for _ in tqdm(
            range(math.ceil(num_samples / batch_size)), desc="Sampling activations"
        ):
            z = torch.randn([batch_size, self.G.z_dim]).cuda()
            img, block_x, block_img = self.G(z, None)
            act = block_x[-1].detach().cpu()
            activations.append(act)
        return torch.cat(activations)[:num_samples]

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.last_conv(activations)

    def target_to_image(self, target: torch.Tensor) -> torch.Tensor:
        return (target + self.last_bias.reshape(1, 3, 1, 1)).clamp(-1, 1)

    def image_to_target(self, image: torch.Tensor) -> torch.Tensor:
        return image - self.last_bias.reshape(1, 3, 1, 1)

    def get_last_linearity(self):
        return self.last_conv, "conv"


if __name__ == "__main__":
    wrapper = StyleGANWrapper(
        checkpoint_path="trained_models/stylegan2/stylegan2-ffhq-256x256.pkl"
    )
    torch.manual_seed(0)
    z = torch.randn([1, wrapper.G.z_dim]).cuda()
    img, block_x, block_img = wrapper.G(z, None)
    act = block_x[-1].detach().cpu()
    act = wrapper.G.synthesis.b256.torgb.affine(act.cuda())
    target = wrapper.act_to_target(act.cuda())
    image = wrapper.target_to_image(target.cuda())
    pass
