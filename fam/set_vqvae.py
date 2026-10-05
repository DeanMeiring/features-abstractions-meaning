"""Set VQ-VAE: write an image as an unordered SET of 8 words that each stand on their own.

    image -> ENCODER (sees the whole image) -> 8 slots -> snap each to nearest WORD
          -> each word draws its own picture layer -> layers are ADDED -> rebuilt image

Grid words (fam/image_vqvae.py) only mean something at their grid position:
"an edge, here". Real words mean the same thing wherever they appear. Here,
the picture layers are added together, and adding doesn't care about order,
so neither slot number nor position can carry meaning. Each word has to say
everything about its own part of the image (what, and where), like
"boot sole" or "dark top half". An image is 8 words = 8 bytes.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fam.image_vqvae import to_tensor


class SetVQVAE(nn.Module):
    def __init__(self, n_words: int = 8, slot_size: int = 16, vocab_size: int = 256):
        super().__init__()
        self.n_words, self.slot_size = n_words, slot_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, n_words * slot_size),             # whole image -> 8 slots
        )
        # One shared drawer: turns ONE word into a picture layer (28x28 scores).
        self.drawer = nn.Sequential(
            nn.Linear(slot_size, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # 14x14 -> 28x28
        )
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)

    def slots(self, x):
        """(n, 1, 28, 28) images -> (n, 8, slot_size) slot vectors."""
        return self.encoder(x).view(len(x), self.n_words, self.slot_size)

    def nearest_words(self, slots):
        flat = slots.reshape(-1, self.slot_size)
        return torch.cdist(flat, self.codebook).argmin(dim=1).view(slots.shape[:-1])

    def layers(self, vectors):
        """(n, 8, slot_size) word vectors -> (n, 8, 28, 28) one picture layer per word."""
        n = len(vectors)
        return self.drawer(vectors.reshape(-1, self.slot_size)).view(n, self.n_words, 28, 28)

    def draw(self, vectors):
        """Add the layers (order can't matter) and squash to brightness 0-1."""
        return torch.sigmoid(self.layers(vectors).sum(dim=1, keepdim=True))

    def decode_words(self, words):
        """(n, 8) word numbers -> (n, 1, 28, 28) rebuilt images."""
        return self.draw(self.codebook[words])


def train(images: np.ndarray, epochs: int = 6, batch_size: int = 128, seed: int = 42) -> SetVQVAE:
    """Learn the vocabulary from images alone (no labels)."""
    torch.manual_seed(seed)
    x_all = to_tensor(images)
    model = SetVQVAE()
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

            # Same three losses as the grid VQ-VAE (see fam/vqvae.py).
            passed_on = slots + (snapped - slots).detach()
            rebuild_loss = F.mse_loss(model.draw(passed_on), x)
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
                model.codebook[dead] = sample[torch.arange(len(dead)), torch.randint(model.n_words, (len(dead),))]
        print(f"  epoch {epoch + 1}/{epochs}: rebuild error {total / len(x_all):.4f}, "
              f"words used {int(used.sum())}/{len(model.codebook)}")
    return model


@torch.no_grad()
def words_of(model: SetVQVAE, images: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Each image as 8 word numbers (uint8, one byte each)."""
    out = [
        model.nearest_words(model.slots(to_tensor(images[i:i + batch_size]))).numpy()
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(out).astype(np.uint8)
