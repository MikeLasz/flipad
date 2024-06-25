import sys
from pathlib import Path
from typing import Any

import torch

# sys.path.append("src")
import logging

from flipad.data_utils import get_test_target

logger = logging.getLogger(__name__)


def remove_paths(paths: list, modelnr: int = None, model: str = None):
    assert (modelnr is not None and model is None) or (
        model is not None and modelnr is None
    ), "exactly one of the arguments modelnr or model must be not None!"
    if modelnr is not None:
        # remove all paths that do not fit to modelnr
        return [
            path
            for path in paths
            if (path[-2:] == "_" + str(modelnr) or path[-2] != "_")
        ]
    else:
        # remove all paths that do not correspond to the same model
        return [path for path in paths if path.startswith(model)]


def sanitize(inp: Any) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(inp))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def float2image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * 127.5 + 128).clamp(0, 255) / 255


def compute_losses(
    wrapper: callable,
    features: torch.tensor,
    avg_features: torch.tensor,
    img_path: Path,
    num_samples: int,
    perturbation: str,
    num_workers: int,
):
    with torch.no_grad():
        # reconstruction loss:
        last_linearity, _ = wrapper.get_last_linearity()
        target_other_test = get_test_target(
            img_path=img_path,
            num_samples=num_samples,
            perturbation=perturbation,
            num_workers=num_workers,
            wrapper=wrapper,
        )
        recon_target_other_test = last_linearity(features.cuda())
        recon_loss = (
            torch.sum(
                (recon_target_other_test - target_other_test.cuda()) ** 2, dim=(1, 2, 3)
            )
            .detach()
            .cpu()
            .numpy()
        )

        # l1-loss
        l1_loss = (
            torch.sum(torch.abs(features - avg_features[None]), dim=(1, 2, 3))
            .detach()
            .cpu()
            .numpy()
        )
    return recon_loss, l1_loss


# Not used but perhaps useful
def compute_reconstruction_losses(linearity, z, output):
    return (
        torch.mean(torch.linalg.norm(linearity(z) - output, axis=1) ** 2, dim=(1, 2))
        .detach()
        .cpu()
        .numpy()
    )


# Not used but perhaps useful
def compute_l2_i_loss(avg_activation, activation):
    return (
        torch.mean(
            torch.linalg.norm(activation - avg_activation, dim=1) ** 2, dim=(1, 2)
        )
        .detach()
        .cpu()
        .numpy()
    )


# Not used but perhaps useful
def compute_l1_i_loss(avg_activation, activation):
    return (
        torch.mean(
            torch.sum(torch.abs(activation - avg_activation), dim=1) ** 2, dim=(1, 2)
        )
        .detach()
        .cpu()
        .numpy()
    )


# Not used but perhaps useful
def compute_inner(avg_act, activation, scaling=True, centering=True):
    with torch.no_grad():
        if centering:
            activation -= torch.mean(activation, dim=(1, 2, 3), keepdim=True)
            avg_act = avg_act - torch.mean(avg_act)
        if scaling:
            activation /= torch.std(activation, dim=(1, 2, 3), keepdim=True)
            avg_act = avg_act / torch.std(avg_act)

        return (
            (torch.flatten(activation, 1, 3) @ avg_act.flatten()).detach().cpu().numpy()
        )
