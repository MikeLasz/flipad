import re
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import wandb

import math


def show(imgs):
    """https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def plot_losses(reconstruction_losses: dict, l1_losses: dict, save_path: Path):
    plt.clf()
    # reconstruction loss:
    fig, ax = plt.subplots()
    for model in reconstruction_losses.keys():
        if reconstruction_losses[model] is not None:
            label = re.sub(r"_[0-9]+", "", model)
            ax.hist(reconstruction_losses[model], label=label, alpha=0.4)

    ax.legend()
    ax.set_xlabel("reconstruction loss")
    ax.set_ylabel("bin counts")
    plt.tight_layout()
    plt.savefig(save_path / "reconstruction_loss.png")
    plt.clf()

    fig, ax = plt.subplots()
    # l1 loss:
    for model in l1_losses.keys():
        if l1_losses[model] is not None:
            label = re.sub(r"_[0-9]+", "", model)
            ax.hist(l1_losses[model], label=label, alpha=0.4)

    ax.legend()
    ax.set_xlabel("l1 distance")
    ax.set_ylabel("bin counts")
    plt.tight_layout()
    plt.savefig(save_path / "l1_loss.png")
    plt.clf()


def plot_tensors(
    tensors: Union[torch.Tensor, List[torch.Tensor]],
    save_path: Path,
    col_labels: List[str] = None,
    row_labels: List[str] = None,
    suptitle: str = None,
    wandb_key: str = "average-activations",
):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    nrows = max(tensor.shape[0] if tensor.ndim == 3 else 1 for tensor in tensors)
    for i in range(len(tensors)):
        if tensors[i].ndim not in [2, 3]:
            raise ValueError("All tensors should have 2 or 3 dimensions.")
        elif tensors[i].ndim == 2:
            tensors[i] = tensors[i].unsqueeze(0)
        elif tensors[i].shape[0] not in [1, nrows]:
            raise ValueError("Incompatible numbers of rows.")

        # repeat single tensors nrows times
        if tensors[i].shape[0] == 1:
            tensors[i] = tensors[i].tile(dims=(nrows, 1, 1))

    if col_labels is not None and len(col_labels) != len(tensors):
        raise ValueError("Different number of tensors and column labels.")

    if row_labels is not None and len(row_labels) != tensors[0].shape[0]:
        raise ValueError("Different number of images and row labels.")

    ncols = len(tensors)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3 * ncols, 3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    max_vals = torch.amax(torch.stack(tensors), dim=(0, 2, 3)).numpy()
    min_vals = torch.amin(torch.stack(tensors), dim=(0, 2, 3)).numpy()
    for col, tensor in enumerate(tensors):
        for row, image in enumerate(tensor):
            axs[row, col].imshow(image, vmin=min_vals[row], vmax=max_vals[row])
            axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if row == 0 and col_labels is not None:
                axs[row, col].set_title(col_labels[col])
            if col == 0 and row_labels is not None:
                axs[row, col].set_ylabel(row_labels[row])
    fig.suptitle(suptitle, fontsize=35)
    fig.savefig(save_path)
    if wandb.run is not None:
        wandb.log({wandb_key: wandb.Image(fig)})


def find_factors_closest_to_sqrt(n):
    """
    Given a number n it returns two factors a and b such that a*b = n and a is as close as possible to sqrt(n).
    This is useful for plotting tabular data in an almost-square grid.
    """
    a = int(math.sqrt(n))
    while a >= 1:
        if n % a == 0:
            b = n // a
            return a, b
        a += 1
    return None, None  # No factors found
