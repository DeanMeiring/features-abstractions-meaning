"""Load Fashion-MNIST: 28x28 greyscale clothing photos in 10 classes."""

import gzip
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "fashion_mnist"
CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read(name: str, header_bytes: int) -> np.ndarray:
    # The files are raw bytes after a small header: one byte per pixel (0-255) or label (0-9).
    with gzip.open(RAW / name) as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=header_bytes)


def load_fashion_mnist(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return images (n, 28, 28) as uint8 and labels (n,). split is 'train' or 't10k'."""
    images = _read(f"{split}-images-idx3-ubyte.gz", 16).reshape(-1, 28, 28)
    labels = _read(f"{split}-labels-idx1-ubyte.gz", 8)
    return images, labels
