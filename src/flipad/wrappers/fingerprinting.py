import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flipad.prnu import extract_single


class FingerprintingWrapper:
    name = "fingerprint"

    def __init__(
        self, num_samples: int = 512, tolerable_fnr: list = [0.0, 0.001, 0.005, 0.01]
    ) -> None:
        self.num_samples = num_samples
        self.fingerprint = None
        self.tolerable_fnr = tolerable_fnr
        self.thresholds = {}

    def train(self, dl: DataLoader):
        residuals = []
        count = 0
        for batch in tqdm(dl, desc="Computing fingerprint"):
            for imgs in batch:
                for img in imgs:
                    residual = extract_single(self.to_numpy(img))
                    residuals.append(residual)
                    count += 1
                    if count == self.num_samples:
                        break
        fingerprint = np.mean(residuals, axis=0)
        self.fingerprint = self.zero_mean_unit_norm(fingerprint)

    def forward(self, image: torch.Tensor) -> float:
        if len(image.shape) == 4:
            corr = []
            for img in image:
                residual = extract_single(self.to_numpy(img))
                corr.append(
                    np.inner(
                        self.fingerprint.flatten(),
                        self.zero_mean_unit_norm(residual).flatten(),
                    )
                )
        else:
            residual = extract_single(self.to_numpy(image))
            corr = np.inner(
                self.fingerprint.flatten(), self.zero_mean_unit_norm(residual).flatten()
            )
        return corr

    @staticmethod
    def to_numpy(image: torch.Tensor) -> np.ndarray:
        if len(image.shape) == 4:
            image = image.squeeze(0)
        elif len(image.shape) == 3:
            pass
        else:
            raise ValueError(f"Image shape {image.shape} is not supported.")
        return ((image.numpy() + 1) * 127.5).astype(np.uint8).transpose((1, 2, 0))

    @staticmethod
    def zero_mean_unit_norm(array: np.ndarray) -> np.ndarray:
        array = array - np.mean(array)
        return array / np.linalg.norm(array)

    def save(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pickle.dump(self.fingerprint, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.fingerprint = pickle.load(f)

    def __repr__(self):
        return f"FingerprintingWrapper(num_samples={self.num_samples})"
