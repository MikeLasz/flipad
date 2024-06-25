import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
from torchvision import transforms


class InceptionLoss(torch.nn.Module):
    """https://pytorch.org/hub/pytorch_vision_inception_v3/"""

    def __init__(self) -> None:
        super().__init__()
        self.inception = (
            torch.hub.load(
                "pytorch/vision:v0.10.0",
                "inception_v3",
                weights="Inception_V3_Weights.DEFAULT",
            )
            .cuda()
            .eval()
        )
        self.preprocess = transforms.Compose(
            [
                # transforms.Resize(299), # Not necessary for our experiments
                # transforms.CenterCrop(299), # Not necessary for our experiments
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 3:
            input = input[None, :, :, :]
        if len(target.shape) == 3:
            target = target[None, :, :, :]
        input = self.preprocess(input)
        target = self.preprocess(target)

        input_features = self.get_features(input)
        target_features = self.get_features(target)

        return torch.nn.functional.mse_loss(input_features, target_features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        return x


class InversionWrapper(torch.nn.Module):
    def __init__(
        self,
        generator_wrapper: callable,
        distance: str,
        lr: float = 0.1,
        num_inits: int = 10,
        num_steps=1000,
        tolerable_fnr: list = [0.0, 0.001, 0.005, 0.01],
        ae: torch.nn.Module = None,
    ) -> None:
        super().__init__()
        if hasattr(generator_wrapper, "forward"):
            self.G = generator_wrapper.forward
        else:
            self.G = generator_wrapper.G
        self.G_name = generator_wrapper.name
        self.seed_dim = generator_wrapper.seed_dim
        self.distance = distance
        self.lr = lr
        self.num_inits = num_inits
        self.num_steps = num_steps
        self.tolerable_fnr = tolerable_fnr
        self.ae = ae

        if distance == "l2":
            self.loss = torch.nn.MSELoss()
        elif distance == "inception":
            self.loss = InceptionLoss()
        else:
            raise NotImplementedError

    def train(self, dl: DataLoader):
        pass

    def forward(self, images: torch.Tensor) -> float:
        images = images.cuda()

        if self.ae is None:
            # Invert self.G by gradient descent on the latents and
            # output smallest reconstruction loss
            differences = []

            for init in tqdm(
                range(self.num_inits),
                desc="Solving Optimization Problem for different Inits",
            ):
                if self.G_name == "medigan_dcgan":
                    seeds = [
                        torch.randn(self.seed_dim, 1, 1).cuda().requires_grad_()
                        for j in range(len(images))
                    ]
                else:
                    seeds = [
                        torch.randn(self.seed_dim).cuda().requires_grad_()
                        for j in range(len(images))
                    ]
                opts = [torch.optim.Adam([seed], lr=self.lr) for seed in seeds]
                for step in tqdm(
                    range(self.num_steps), desc="Solving Optimization Problem"
                ):
                    [opt.zero_grad() for opt in opts]
                    output = self.G(torch.stack(seeds))
                    loss = self.loss(images, output)
                    loss.backward()
                    # for l in loss:
                    #    l.mean().backward(retain_graph=True)
                    [opt.step() for opt in opts]

                # compute final reconstruction loss:
                final_losses = []
                with torch.no_grad():
                    outputs = self.G(torch.stack(seeds))
                    for j in range(len(images)):
                        image = images[j]
                        final_losses.append(self.loss(image, outputs[j]))
                    differences.append(final_losses)

            min_distances = (
                torch.tensor(differences).min(dim=0).values.detach().cpu().numpy()
            )
            return min_distances
        else:
            generator = torch.Generator(device="cuda").manual_seed(42)
            self.ae.enable_slicing()
            self.ae.cuda()
            with torch.no_grad():
                reconstruction_losses = []
                for init in tqdm(
                    range(self.num_inits),
                    desc="Computing reconstruction loss for different inits",
                ):
                    images = (images + 1) / 2
                    reconstruction_losses_init = []
                    latents = self.ae.encode(images).latent_dist.sample(
                        generator=generator
                    )
                    reconstructions_init = self.ae.decode(latents).sample.clamp(0, 1)

                    for img, recon in zip(images, reconstructions_init):
                        reconstruction_losses_init.append(self.loss(img, recon)[None])
                    reconstruction_losses_init = torch.concat(
                        reconstruction_losses_init
                    )
                    reconstruction_losses.append(reconstruction_losses_init[None])

                # Select the best init per image:
                reconstruction_losses = torch.concat(reconstruction_losses)
                best_reconstruction_losses = torch.min(
                    reconstruction_losses, dim=0
                ).values
            return best_reconstruction_losses.detach().cpu().numpy()

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass

    def __repr__(self):
        return f"InversionWrapper(distance={self.distance},lr={self.lr},num_inits={self.num_inits},num_steps={self.num_steps})"


class L2InversionWrapper(InversionWrapper):
    name = "l2_inversion"

    def __init__(
        self,
        generator_wrapper: callable,
        lr: float = 0.1,
        num_inits: int = 10,
        num_steps=1000,
        ae=None,
    ) -> None:
        super().__init__(
            generator_wrapper=generator_wrapper,
            distance="l2",
            lr=lr,
            num_inits=num_inits,
            num_steps=num_steps,
            ae=ae,
        )


class InceptionInversionWrapper(InversionWrapper):
    name = "inception_inversion"

    def __init__(
        self,
        generator_wrapper: callable,
        lr: float = 0.1,
        num_inits: int = 10,
        num_steps=1000,
        ae=None,
    ) -> None:
        super().__init__(
            generator_wrapper=generator_wrapper,
            distance="inception",
            lr=lr,
            num_inits=num_inits,
            num_steps=num_steps,
            ae=ae,
        )
