from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms.functional import center_crop, normalize, resize, to_tensor

from sad.base import TorchvisionDataset


class DeepSADDataset(TorchvisionDataset):
    def __init__(self, X_train, X_test, X_val, y_train, y_test):
        self.n_classes = 2  # 0: normal, 1: outlier

        self.train_set = TensorDataset(
            X_train,
            y_train,
            y_train,
            torch.arange(len(y_train)),
        )
        self.test_set = TensorDataset(
            X_test,
            y_test,
            torch.zeros_like(y_test),
            torch.arange(len(y_test)),
        )

        self.val_set = TensorDataset(
            X_val,
            torch.ones(len(X_val)),
            torch.ones(len(X_val)),
            torch.arange(len(X_val)),
        )


class ImageFolder(VisionDataset):
    def __init__(
        self,
        root: Path,
        num_images: Optional[int] = None,
        transform: Optional[callable] = None,
        resize_to: Optional[int] = None,
        color_space: str = "RGB",
    ) -> None:
        super().__init__(root)
        assert color_space in ["RGB", "L"]
        self.color_space = color_space
        self.img_paths = []
        for path in sorted(root.iterdir()):
            if path.suffix in IMG_EXTENSIONS:
                if not "mask" in str(path):  # ignore masks in medical datasets
                    self.img_paths.append(path)
            if num_images is not None and len(self.img_paths) >= num_images:
                break
        if num_images is not None and len(self.img_paths) < num_images:
            raise ValueError(
                f"{root} contains not enough images ({len(self.img_paths)} / {num_images})."
            )
        self.transform = transform
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.img_paths[idx]).convert(self.color_space)
        if self.resize_to is not None and img.size != (self.resize_to, self.resize_to):
            img = resize(img, size=self.resize_to)
            img = center_crop(img, output_size=self.resize_to)
        if self.transform is not None:
            img = self.transform(img)
        img = to_tensor(img)
        img = normalize(img, mean=[0.5], std=[0.5])
        return img, str(self.img_paths[idx])


def get_data_loader(
    batch_size, root="./data/", data="mnist", image_size=64, num_workers=4, shuffle=True
):
    if data == "mnist":
        train_dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
            download=True,
        )
    elif data == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            download=True,
        )
    elif data == "celeba":
        train_dataset = datasets.CelebA(
            root=root,
            split="train",
            download=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif data == "lsun":
        train_dataset = datasets.LSUN(
            root=root,
            classes=["bedroom_train"],
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    else:
        raise NotImplementedError(f"Dataset {data} is not supported.")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return train_loader


def load_images(path: Path, num_images: int, num_workers: int) -> torch.Tensor:
    ds = ImageFolder(root=path, num_images=num_images)
    dl = DataLoader(dataset=ds, batch_size=num_images, num_workers=num_workers)
    return next(iter(dl))


def load_dataset(
    path_class1: Path,
    path_class2: Path,
    perturbation,
    sample_label: bool,
    num_samples: int,
    num_workers: int,
    resize_to: int,
):
    y = []
    imgs = []
    data_type = (
        "tabular"
        if (
            "whitewine" in str(path_class1)
            or "redwine" in str(path_class1)
            or "parkinsons" in str(path_class1)
        )
        else "image"
    )
    for path in [path_class1, path_class2]:
        if path != "":
            if data_type == "tabular":
                data = torch.tensor(np.genfromtxt(path / "data.csv", delimiter=","))[
                    :num_samples
                ].to(torch.float32)
                dl = DataLoader(
                    dataset=data,
                    batch_size=128,
                    num_workers=num_workers,
                )
            else:
                ds = ImageFolder(
                    root=path,
                    num_images=num_samples,
                    transform=perturbation,
                    resize_to=resize_to,
                )
                dl = DataLoader(
                    dataset=ds,
                    batch_size=128,
                    num_workers=num_workers,
                )

            label = 1 if path == path_class1 else 0
            for batch in dl:
                if isinstance(batch, list):
                    imgs.append(batch[0])
                else:
                    imgs.append(batch)

                y.append(torch.tensor([label] * len(batch)))

    imgs = torch.cat(imgs)
    y = torch.cat(y)
    if sample_label:
        ds = TensorDataset(imgs, y)
    else:
        ds = TensorDataset(imgs)
    return ds


def get_test_target(
    img_path: Path,
    num_samples: int,
    perturbation: Optional[callable],
    num_workers: int,
    wrapper: callable,
) -> torch.Tensor:
    ds = ImageFolder(root=img_path, num_images=num_samples, transform=perturbation)
    dl = DataLoader(dataset=ds, batch_size=num_samples, num_workers=num_workers)
    test_image, _ = next(iter(dl))
    test_target = wrapper.image_to_target(test_image)
    return test_target
