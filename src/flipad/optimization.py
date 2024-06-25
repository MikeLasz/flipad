import logging
import pickle
from pathlib import Path
from typing import Optional

import torch
from torch.nn.functional import interpolate, max_pool2d
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch_dct import dct_2d
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

import numpy as np

from flipad.data_utils import ImageFolder
from lasso.conv2d.ista import ista_conv2d
from lasso.linear.solvers import ista
from flipad.utils import compute_losses, sanitize

logger = logging.getLogger(__name__)


def reconstruct_activations(
    img_path: Path,
    num_samples: int,
    perturbation: Optional[callable],
    wrapper: callable,
    reg_anchor: torch.Tensor,
    alpha: float,
    momentum: float,
    lr: float,
    max_iter: int,
    num_workers: int,
    cache_path: Path,
    compute_loss: bool = False,
    downsampling: str = "avg",
    avoid_caching: bool = False,
) -> torch.Tensor:
    logger.info(
        f"Reconstructing activations for {img_path} with perturbation {perturbation}"
    )
    if wrapper.name == "stablediffusion":
        resize_img_to = 512
        resize_act_to = 128
        batch_size = 32
    elif wrapper.name == "medigan_dcgan":
        resize_img_to = None
        resize_act_to = 64
        batch_size = 128
    else:
        resize_img_to = None
        resize_act_to = 32
        batch_size = min(1000, num_samples)

    # get reconstructions
    if perturbation is None:
        perturbation_str = "clean"
    else:
        perturbation_str = sanitize(str(perturbation))

    reconstruction_path = (
        cache_path
        / f"reconstructions_{downsampling}"
        / img_path
        / f"model={wrapper.name}-checkpoint={sanitize(wrapper.checkpoint_path)}-num_samples={num_samples}-alpha={alpha}-momentum={momentum}-lr={lr}-max_iter={max_iter}"
        / f"{perturbation_str}.pkl"
    )
    losses_path = (
        cache_path
        / f"losses"
        / img_path
        / f"model={wrapper.name}-checkpoint={sanitize(wrapper.checkpoint_path)}-num_samples={num_samples}-alpha={alpha}-momentum={momentum}-lr={lr}-max_iter={max_iter}"
        / f"{perturbation_str}.pkl"
    )
    rec_losses, l1_losses = None, None
    if reconstruction_path.exists():
        logger.info(f"Loading cached reconstructions from {reconstruction_path}.")
        with open(reconstruction_path, "rb") as f:
            reconstructions = pickle.load(f)
        if losses_path.exists():
            logger.info(f"Loading cached losses from {losses_path}.")
            with open(losses_path, "rb") as f:
                rec_losses, l1_losses = pickle.load(f)
    else:
        color_space = (
            "L" if "medigan" in wrapper.name else "RGB"
        )  # medical data is greyscale (L)
        ds = ImageFolder(
            root=img_path,
            num_images=num_samples,
            transform=perturbation,
            resize_to=resize_img_to,
            color_space=color_space,
        )
        dl = DataLoader(dataset=ds, batch_size=batch_size, num_workers=num_workers)
        reconstructions = []
        for batch, path in tqdm(dl, desc=f"Reconstructing activations for {img_path}"):
            batch = wrapper.image_to_target(batch.cuda())

            rec = fista_reconstruct(
                output=batch,
                wrapper=wrapper,
                reg_anchor=reg_anchor,
                alpha=alpha,
                lr=lr,
                max_iter=max_iter,
            )
            logger.info(
                f"avg-similarity of the reconstruction: {torch.sum(rec.cuda() == reg_anchor.cuda()) / rec.numel()}"
            )

            if wrapper.name in ["dcgan", "wgangp", "ebgan"]:
                # no downsampling necessary
                reconstructions.append(rec)
            elif wrapper.name == "medigan_dcgan":
                # for now: No downsampling
                reconstructions.append(rec)
            elif wrapper.name == "lsgan":
                if downsampling == "avg":
                    reconstructions.append(
                        interpolate(rec, size=(resize_act_to, resize_act_to))
                    )
                elif downsampling == "max":
                    kernel_size = 2
                    reconstructions.append(
                        max_pool2d(rec, kernel_size=kernel_size, stride=kernel_size)
                    )  # 64x64x64 -> 64x32x32
                elif downsampling == "center_crop":
                    reconstructions.append(
                        center_crop(rec, (resize_act_to, resize_act_to))
                    )
            elif wrapper.name == "stablediffusion":
                if downsampling == "avg":
                    reconstructions.append(
                        interpolate(rec, size=(resize_act_to, resize_act_to))
                    )
                elif downsampling == "max":
                    kernel_size = 16
                    reconstructions.append(
                        max_pool2d(rec, kernel_size=kernel_size, stride=kernel_size)
                    )  # 128x512x512 -> 128x128x128
                elif downsampling == "center_crop":
                    reconstructions.append(
                        center_crop(rec, (resize_act_to, resize_act_to))
                    )
                elif downsampling == "random_crop":
                    crop = RandomCrop(resize_act_to)
                    reconstructions.append(crop(rec))
                elif downsampling == "none":
                    pass
        reconstructions = torch.cat(reconstructions)
        if not avoid_caching:
            reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
            with open(reconstruction_path, "wb") as f:
                pickle.dump(reconstructions, f)
        rec_losses, l1_losses = None, None
        if compute_loss:
            # compute reconstruction and l1-loss
            rec_losses, l1_losses = compute_losses(
                wrapper=wrapper,
                features=rec,
                avg_features=reg_anchor,
                img_path=img_path,
                num_samples=batch_size,
                perturbation=perturbation,
                num_workers=num_workers,
            )
            losses_path.parent.mkdir(parents=True, exist_ok=True)
            with open(losses_path, "wb") as f:
                pickle.dump([rec_losses, l1_losses], f)

    return [reconstructions, rec_losses, l1_losses]


def reconstruct_activations_tabular(
    path: Path,
    num_samples: int,
    perturbation: Optional[callable],
    wrapper: callable,
    reg_anchor: torch.Tensor,
    alpha: float,
    momentum: float,
    lr: float,
    max_iter: int,
    num_workers: int,
    cache_path: Path,
    compute_loss: bool = False,
    downsampling: str = "avg",
    reference_dl=None,
    avoid_caching: bool = False,
) -> torch.Tensor:
    logger.info(
        f"Reconstructing activations for {path} with perturbation {perturbation}"
    )
    batch_size = 1000

    # get reconstructions
    if perturbation is None:
        perturbation_str = "clean"
    else:
        perturbation_str = sanitize(str(perturbation))

    reconstruction_path = (
        cache_path
        / f"reconstructions_{downsampling}"
        / path
        / f"model={wrapper.name}-checkpoint={sanitize(wrapper.checkpoint_path)}-num_samples={num_samples}-alpha={alpha}-momentum={momentum}-lr={lr}-max_iter={max_iter}"
        / f"{perturbation_str}.pkl"
    )
    losses_path = (
        cache_path
        / f"losses"
        / path
        / f"model={wrapper.name}-checkpoint={sanitize(wrapper.checkpoint_path)}-num_samples={num_samples}-alpha={alpha}-momentum={momentum}-lr={lr}-max_iter={max_iter}"
        / f"{perturbation_str}.pkl"
    )
    rec_losses, l1_losses = None, None
    if reconstruction_path.exists():
        logger.info(f"Loading cached reconstructions from {reconstruction_path}.")
        with open(reconstruction_path, "rb") as f:
            reconstructions = pickle.load(f)
        if losses_path.exists():
            logger.info(f"Loading cached losses from {losses_path}.")
            with open(losses_path, "rb") as f:
                rec_losses, l1_losses = pickle.load(f)
        assert (
            len(reconstructions) == num_samples
        ), f"Only {len(reconstructions)} samples in {path}, but {num_samples} requested."
    else:
        data = torch.tensor(np.genfromtxt(path / "data.csv", delimiter=","))
        assert (
            len(data) >= num_samples
        ), f"Only {len(data)} samples in {path}, but {num_samples} requested."
        data = data[:num_samples].to(torch.float32)
        dl = DataLoader(
            dataset=data,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        reconstructions = []
        for batch in tqdm(dl, desc=f"Reconstructing activations for {path}"):
            batch = wrapper.image_to_target(batch.cuda())
            rec = fista_reconstruct(
                output=batch,
                wrapper=wrapper,
                reg_anchor=reg_anchor,
                alpha=alpha,
                lr=lr,
                max_iter=max_iter,
            )
            logger.info(
                f"avg-similarity of the reconstruction: {torch.sum(rec.cuda() == reg_anchor.cuda()) / rec.numel()}"
            )
            reconstructions.append(rec)

        reconstructions = torch.cat(reconstructions)
        if not avoid_caching:
            reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
            with open(reconstruction_path, "wb") as f:
                pickle.dump(reconstructions, f)
        rec_losses, l1_losses = None, None
        if compute_loss:
            # compute reconstruction and l1-loss
            rec_losses, l1_losses = compute_losses(
                wrapper=wrapper,
                features=rec,
                avg_features=reg_anchor,
                img_path=path,
                num_samples=batch_size,
                perturbation=perturbation,
                num_workers=num_workers,
            )
            losses_path.parent.mkdir(parents=True, exist_ok=True)
            with open(losses_path, "wb") as f:
                pickle.dump([rec_losses, l1_losses], f)

    return [reconstructions, rec_losses, l1_losses]


def select_channels(data: torch.tensor, num_samples: int, select_channels: int = 3):
    avg1 = torch.mean(data[:num_samples], dim=0)
    avg2 = torch.mean(data[num_samples:], dim=0)

    # maximum distance on average
    difference = torch.mean(torch.abs(avg1 - avg2), dim=(1, 2))
    channels = torch.topk(difference, select_channels)[1]
    return channels.numpy()


def select_dimensions(data: torch.tensor, num_samples: int, select_dimensions: int = 3):
    avg1 = torch.mean(data[:num_samples], dim=0)
    avg2 = torch.mean(data[num_samples:], dim=0)

    # maximum distance on average
    difference = torch.abs(avg1 - avg2)
    dimensions = torch.topk(difference, select_dimensions)[1]
    return dimensions.numpy()


def find_closest_activation(
    reference_dl,
    data: torch.Tensor,
    data_path: Path,
    destination_path: Path,  # where we save the path to the closest activation
    distance: callable,
):
    destination_path.mkdir(exist_ok=True, parents=True)
    act = []
    for j in tqdm(range(len(data)), desc="Finding the closest activation in batch..."):
        sample = data[j]
        path = Path(data_path[j])
        save_path = destination_path / f'{path.name.split(".")[0]}.txt'
        if save_path.exists():
            with open(save_path, "r") as f:
                act_path = f.read()
            act.append(torch.load(act_path)[None])
        else:
            min_distance = torch.inf
            for reference_batch, reference_path in reference_dl:
                reference_batch = reference_batch.cuda()
                distances = distance(sample, reference_batch)
                distances[distances == 0] = (
                    torch.inf
                )  # <- relevant for training fakes: we dont want to select the same activation
                min_distance_in_batch = torch.min(distances)
                if min_distance_in_batch < min_distance:
                    min_distance = min_distance_in_batch
                    min_path = reference_path[torch.argmin(distances)]
            with open(save_path, "w") as f:
                # write the path to the relevant activation
                act_name = Path(min_path).name.split(".")[0]  # gives the number: 005321
                act_path = str(
                    Path(*min_path.split("/")[:-2]) / "act" / f"{act_name}.pt"
                )
                f.write(act_path)
            act.append(torch.load(act_path)[None])
    return torch.cat(act)


def l2_distance(target: torch.Tensor, references: torch.Tensor):
    # returns the scaled squared l2-distance between target and each entry in references
    # Note: squared l2-distance = sum( (target-ref)**2) proportional to mean((target-ref)**2)
    assert len(target.shape) == (
        len(references.shape) - 1
    ), f"references must be a batch of samples with shape {target.shape}."
    target = target[None]
    return torch.mean((target - references) ** 2, dim=(1, 2, 3))


def baseline_features(
    img_path: Path,
    num_samples: int,
    perturbation: callable,
    transform: str,
    batch_size: int,
    num_workers: int,
    resize_to=None,
    downsampling: str = "avg",
    compute_loss: bool = False,  # To make function design consistent
) -> torch.Tensor:
    logger.info(
        f"Computing {transform} features for {img_path} with perturbation {perturbation}"
    )
    color_space = "L" if "bcdr" in str(img_path) else "RGB"  # bcdr images are grayscale

    if any(name in str(img_path) for name in ["coco"]):
        # coco samples are not square images, i.e. they need to be reshaped to 512x512 before applying dct.
        ds = ImageFolder(
            root=img_path,
            num_images=num_samples,
            transform=perturbation,
            resize_to=512,
            color_space=color_space,
        )
    else:
        ds = ImageFolder(
            root=img_path,
            num_images=num_samples,
            transform=perturbation,
            color_space=color_space,
        )
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    features = []
    for batch, _ in tqdm(
        dl,
        desc=f"Computing baseline features from {img_path} with transform {transform}",
    ):
        if resize_to is not None:
            assert isinstance(
                resize_to, int
            ), "Resize to must be an integer or None if no resizing desired."
            if downsampling == "avg":
                batch = interpolate(batch, size=(resize_to, resize_to))
            elif downsampling == "max":
                kernel_size = int(batch.shape[2] / resize_to) ** 2
                batch = max_pool2d(batch, kernel_size=kernel_size, stride=kernel_size)
            elif downsampling == "center_crop":
                batch = center_crop(batch, (resize_to, resize_to))
            elif downsampling == "random_crop":
                crop = RandomCrop(resize_to)
                batch = crop(batch)  # random_crop(batch, (resize_to, resize_to))

        if transform == "dct":
            batch = dct(batch)
        elif transform == "raw":
            pass
        else:
            NotImplementedError

        features.append(batch)
    return [torch.cat(features), None, None]  # return None, None for API consistency


def baseline_features_tabular(
    path: Path,
    num_samples: int,
    perturbation: callable,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    logger.info(f"Computing features for {path} with perturbation {perturbation}")

    data = torch.tensor(np.genfromtxt(path / "data.csv", delimiter=","))[
        :num_samples
    ].to(torch.float32)
    dl = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    features = []
    for batch in tqdm(
        dl,
        desc=f"Computing baseline features from {path}",
    ):

        features.append(batch)
    return [torch.cat(features), None, None]  # return None, None for API consistency


def dct(tensor: torch.Tensor, eps=1e-10) -> torch.Tensor:
    return torch.log(torch.abs(dct_2d(tensor)) + eps)


def preprocess_and_dct(
    tensor: torch.Tensor,
    labels: torch.Tensor = None,
    std=None,
    mean=None,
    eps_dct=1e-10,
    eps_std=0.05,
) -> torch.Tensor:
    if std is None:
        std = tensor[labels == 1].std(dim=0)
    if mean is None:
        mean = tensor[labels == 1].mean(dim=0)
    tensor = (tensor - mean) / (std + eps_std)
    return dct(tensor, eps=eps_dct)


def fista_reconstruct(
    output: torch.Tensor,
    wrapper: callable,
    reg_anchor: torch.Tensor,
    max_iter: int = 10000,
    lr: float = 0.00025,
    alpha: float = 0.0,
) -> torch.Tensor:
    output = output.cuda()
    reg_anchor = reg_anchor.cuda()
    try:
        last_linearity, linearity_type = wrapper.get_last_linearity()
    except NotImplementedError:
        print("wrapper.get_last_linearity() not implemented.")
    if len(reg_anchor.shape) == 3:
        reg_anchor = reg_anchor[None]
        fista_init = torch.cat(len(output) * [reg_anchor])
    elif len(reg_anchor.shape) == 4:
        fista_init = reg_anchor
    else:
        fista_init = torch.cat(len(output) * [reg_anchor[None]])

    with torch.no_grad():
        output_prime = output - last_linearity(
            reg_anchor
        )  # = output - bias - (last_linearity(avg) - bias)
        if linearity_type in ["conv", "tconv"]:
            act_prime = ista_conv2d(
                linearity_type,
                output_prime,
                fista_init,
                last_linearity.weight.data,
                stride=last_linearity.stride,
                padding=last_linearity.padding,
                lr=lr,
                alpha=alpha,
                maxiter=max_iter,
                verbose=True,
                logger=logger,
            )
        elif linearity_type == "fc":
            act_prime = ista(
                output_prime.to(torch.float32),
                fista_init,
                last_linearity.weight.data,
                lr=lr,
                alpha=alpha,
                maxiter=max_iter,
                verbose=True,
                logger=logger,
            )
        else:
            raise NotImplementedError
        act = act_prime + reg_anchor
    return act.detach().cpu()


def sgd_reconstruct(
    output: torch.Tensor,
    wrapper: callable,
    avg_act: torch.Tensor,
    max_iter: int = 50000,
    lr: float = 1e-2,
    momentum: float = 0.0,
    alpha: float = 0.0,
    es_limit: int = 10,
    es_delta: float = 0.0001,
) -> torch.Tensor:
    output = output.cuda()
    avg_act = avg_act.cuda()
    act = torch.stack(len(output) * [avg_act]).requires_grad_()
    optimizer = SGD(params=[act], lr=lr, momentum=momentum)
    n_iter = 0
    es_counter = 0
    loss_prev = torch.inf
    best_act = None

    pbar = tqdm(total=max_iter)
    while True:
        # optimization step
        optimizer.zero_grad()
        loss = torch.linalg.norm(
            (wrapper.act_to_target(act) - output).reshape((len(output), -1)), dim=1
        ) ** 2 + alpha * torch.sum(torch.abs(act - avg_act), dim=(1, 2, 3))
        loss = loss.mean()

        # early stopping
        if loss_prev - loss.item() < es_delta:
            if es_counter == 0:
                best_act = act.clone().detach().cpu()
            es_counter += 1
            if es_counter == es_limit:
                break
        else:
            es_counter = 0
            loss_prev = loss.item()

        loss.backward()
        optimizer.step()

        # check for max iterations
        n_iter += 1
        if n_iter >= max_iter:
            best_act = act.detach().cpu()
            break

        pbar.set_description(
            f"Loss: {loss.item() / len(output):.5f}, ES: {es_counter} / {es_limit}"
        )
        pbar.update()

    logger.info(str(pbar))
    return best_act


def compute_avg_activation(wrapper, num_samples: int, batch_size: int) -> torch.Tensor:
    with torch.no_grad():
        avg_act = torch.mean(
            wrapper.sample_activations(num_samples=num_samples, batch_size=batch_size),
            dim=0,
            dtype=torch.get_default_dtype(),
        )
        return avg_act
