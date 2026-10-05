"""VQ-VAE for images: write a 28x28 picture as a 7x7 grid of words.

    image (784 pixels) -> ENCODER -> 7x7 grid of slots -> snap each to nearest WORD -> DECODER -> rebuilt image

Same idea as fam/vqvae.py, but the encoder uses convolutions: small filters
that slide over the image and respond to local shapes (edges, curves,
textures). Two steps of halving the size (28 -> 14 -> 7) leave a 7x7 grid,
where each cell describes a 4x4 patch of the picture. Each cell is snapped
to a word, so an image becomes 49 words, i.e. 49 bytes instead of 784.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ImageVQVAE(nn.Module):
    def __init__(self, slot_size: int = 16, vocab_size: int = 256):
        super().__init__()
        self.slot_size = slot_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, slot_size, kernel_size=1),                # one slot per grid cell
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
            nn.Sigmoid(),  # pixel brightness between 0 and 1
        )
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)

    def slots(self, x):
        """(n, 1, 28, 28) images -> (n, 7, 7, slot_size) slot vectors."""
        return self.encoder(x).permute(0, 2, 3, 1)

    def nearest_words(self, slots):
        flat = slots.reshape(-1, self.slot_size)
        return torch.cdist(flat, self.codebook).argmin(dim=1).view(slots.shape[:-1])

    def decode_words(self, words):
        """(n, 7, 7) word numbers -> (n, 1, 28, 28) rebuilt images."""
        return self.decoder(self.codebook[words].permute(0, 3, 1, 2))


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (n, 28, 28) -> float (n, 1, 28, 28) between 0 and 1."""
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0


def train(images: np.ndarray, epochs: int = 5, batch_size: int = 128, seed: int = 42) -> ImageVQVAE:
    """Learn the vocabulary from images alone (no labels)."""
    torch.manual_seed(seed)
    x_all = to_tensor(images)
    model = ImageVQVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs):
        order = torch.randperm(len(x_all))
        used = torch.zeros(len(model.codebook), dtype=torch.bool)
        total = 0.0
        for start in range(0, len(x_all), batch_size):
            x = x_all[order[start:start + batch_size]]
            slots = model.slots(x)
            words = model.nearest_words(slots)
            snapped = model.codebook[words]
            used[words.flatten()] = True

            # Straight-through trick, vocab loss and commit loss: see fam/vqvae.py.
            passed_on = slots + (snapped - slots).detach()
            rebuilt = model.decoder(passed_on.permute(0, 3, 1, 2))
            rebuild_loss = F.mse_loss(rebuilt, x)
            loss = (
                rebuild_loss
                + F.mse_loss(snapped, slots.detach())
                + 0.25 * F.mse_loss(slots, snapped.detach())
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += rebuild_loss.item() * len(x)

        # Revive words nobody used this epoch by moving them onto real slots.
        dead = (~used).nonzero().flatten()
        if len(dead) and epoch < epochs - 1:
            with torch.no_grad():
                sample = model.slots(x_all[torch.randint(len(x_all), (len(dead),))])
                model.codebook[dead] = sample[:, 3, 3]  # a slot from the middle of each image
        print(f"  epoch {epoch + 1}/{epochs}: rebuild error {total / len(x_all):.4f}, "
              f"words used {int(used.sum())}/{len(model.codebook)}")
    return model


@torch.no_grad()
def words_of(model: ImageVQVAE, images: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Each image as a 7x7 grid of word numbers (uint8, one byte each)."""
    out = [
        model.nearest_words(model.slots(to_tensor(images[i:i + batch_size]))).numpy()
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(out).astype(np.uint8)
