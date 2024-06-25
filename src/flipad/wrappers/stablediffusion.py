import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from torchvision.utils import save_image

logger = logging.getLogger(__name__)


class StableDiffusionWrapper:
    name = "stablediffusion"
    seed_dim = (4, 64, 64)

    def __init__(self, checkpoint_path: Path):
        logger.info(f"Loading checkpoint {checkpoint_path}.")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(checkpoint_path), safety_checker=None
        )

        self.conv_out = pipe.components["vae"].decoder.conv_out.cuda()
        self.checkpoint_path = checkpoint_path
        self.G = pipe.components["vae"].decoder.cuda()
        self.ae = pipe.vae

    def sample_activations(
        self, num_samples: int, batch_size: int = 1
    ) -> torch.Tensor:  # we do not actually sample but load precomputed file
        model_name = self.checkpoint_path.parts[-1]
        return torch.load(
            f"data/coco2014train/{model_name}/avg_act/avg_act-{model_name}-samples={num_samples}.pt"
        ).unsqueeze(0)

    def act_to_target(self, activations: torch.Tensor) -> torch.Tensor:
        return self.conv_out(activations)

    def target_to_image(self, target: torch.Tensor) -> torch.Tensor:
        return target.clamp(-1, 1)

    def image_to_target(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def get_last_linearity(self):
        return self.conv_out, "conv"
