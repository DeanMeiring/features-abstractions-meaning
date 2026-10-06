"""Window words: write a 24-hour history of telecom activity as an unordered set of 8 words.

    24 hours x 3 activities (72 numbers) -> ENCODER -> 8 slots -> snap each to nearest WORD (256)
        -> each word draws its own 72-number layer -> layers ADDED -> rebuilt history

Experiment 10's set-word recipe (fam/set_vqvae.py), with an encoder for 72
numbers instead of a 28x28 image: the layers are added, so the words are an
unordered set and each must stand on its own. Trained only to rebuild the
history (no future hours), on training-period windows only.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WindowWords(nn.Module):
    def __init__(self, size: int = 72, n_words: int = 8, slot_size: int = 16, vocab_size: int = 256):
        super().__init__()
        self.size, self.n_words, self.slot_size = size, n_words, slot_size
        self.encoder = nn.Sequential(
            nn.Linear(size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_words * slot_size),
        )
        self.drawer = nn.Sequential(nn.Linear(slot_size, 64), nn.ReLU(), nn.Linear(64, size))  # one word -> one layer
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)

    def slots(self, x):
        """(n, 72) standardised history -> (n, 8, slot_size)."""
        return self.encoder(x).view(len(x), self.n_words, self.slot_size)

    def nearest_words(self, slots):
        flat = slots.reshape(-1, self.slot_size)
        return torch.cdist(flat, self.codebook).argmin(dim=1).view(slots.shape[:-1])

    def draw(self, vectors):
        """Add the 8 layers: (n, 8, slot_size) -> (n, 72)."""
        n = len(vectors)
        return self.drawer(vectors.reshape(-1, self.slot_size)).view(n, self.n_words, self.size).sum(dim=1)


def train(windows: np.ndarray, epochs: int = 6, batch_size: int = 1024, seed: int = 0) -> WindowWords:
    """Learn the dictionary from standardised training windows alone (no targets)."""
    torch.manual_seed(seed)
    x_all = torch.tensor(windows, dtype=torch.float32)
    model = WindowWords(size=windows.shape[1])
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
            # Same three losses as the other VQ-VAEs (see fam/vqvae.py).
            passed_on = slots + (snapped - slots).detach()
            rebuild_loss = F.mse_loss(model.draw(passed_on), x)
            loss = rebuild_loss + F.mse_loss(snapped, slots.detach()) + 0.25 * F.mse_loss(slots, snapped.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += rebuild_loss.item() * len(x)
        dead = (~used).nonzero().flatten()
        if len(dead) and epoch < epochs - 1:  # revive unused words on real slots
            with torch.no_grad():
                sample = model.slots(x_all[torch.randint(len(x_all), (len(dead),))])
                model.codebook[dead] = sample[torch.arange(len(dead)), torch.randint(model.n_words, (len(dead),))]
        print(f"  epoch {epoch + 1}/{epochs}: rebuild error {total / len(x_all):.4f}, words used {int(used.sum())}/256")
    return model.eval()


@torch.no_grad()
def words_of(model: WindowWords, windows: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    """(n, 72) standardised histories -> (n, 8) word numbers (uint8)."""
    out = [model.nearest_words(model.slots(torch.tensor(windows[i:i + batch_size], dtype=torch.float32))).numpy()
           for i in range(0, len(windows), batch_size)]
    return np.concatenate(out).astype(np.uint8)
