import io

import numpy as np
from PIL import Image
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import center_crop, normalize, resize


class GaussianNoise:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, image: Image):
        arr = np.asarray(image)
        noise = (
            np.random.default_rng().normal(loc=0, scale=self.std, size=arr.shape) * 255
        )
        return Image.fromarray((arr + noise).round().clip(0, 255).astype(np.uint8))

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std})"


class JPEGCompression:
    def __init__(self, quality: float) -> None:
        self.quality = int(quality)

    def __call__(self, image: Image.Image):
        with io.BytesIO() as f:
            image.save(f, format="JPEG", quality=self.quality)
            img = Image.open(f)
            img.load()
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(quality={self.quality})"


class ResizedCrop:
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, image: Image.Image):
        return resize(center_crop(image, self.size), image.height)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(size={self.size})"


class MyGaussianBlur(GaussianBlur):
    def __init__(self, size: int) -> None:
        super().__init__(kernel_size=size, sigma=3)
        self.size = size

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(size={self.size})"
