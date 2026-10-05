"""Words of words: compress a 7x7 grid of level-1 words into a 4x4 grid of level-2 words.

    level-1 words (7x7) -> look up their vectors -> ENCODER -> 4x4 slots
        -> snap each to nearest LEVEL-2 WORD -> DECODER -> guess the 49 level-1 words

Level-1 words (fam/image_vqvae.py) each describe a 4x4 pixel patch ("edge
here"). This model never sees pixels: it reads the level-1 words and learns
which combinations of neighbouring words keep turning up together. Each
level-2 slot looks at a 3x3 block of level-1 words (about a quarter of the
image), so a level-2 word can stand for a bigger shape ("heel", "collar").

How it learns, without labels: from only 16 level-2 words it must rebuild
which of the 256 level-1 words sat in each of the 49 cells. That is a
256-way guess per cell, so the loss is cross-entropy (how wrong the guesses
are), not the pixel error used at level 1.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WordVQVAE(nn.Module):
    def __init__(self, level1_codebook: torch.Tensor, slot_size: int = 32, vocab_size: int = 256):
        super().__init__()
        # Level-1 word vectors are fixed: level 2 builds on level 1, it must not change it.
        self.register_buffer("level1", level1_codebook.detach().clone())
        level1_vocab, level1_size = level1_codebook.shape
        self.slot_size = slot_size
        self.encoder = nn.Sequential(
            nn.Conv2d(level1_size, 64, kernel_size=3, padding=1),        # mix neighbouring words
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),        # 7x7 -> 4x4
            nn.ReLU(),
            nn.Conv2d(64, slot_size, kernel_size=1),                      # one slot per cell
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),  # 4x4 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, level1_vocab, kernel_size=1),  # a score for each level-1 word, per cell
        )
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)

    def slots(self, words1):
        """(n, 7, 7) level-1 word numbers -> (n, 4, 4, slot_size) slot vectors."""
        x = self.level1[words1].permute(0, 3, 1, 2)
        return self.encoder(x).permute(0, 2, 3, 1)

    def nearest_words(self, slots):
        flat = slots.reshape(-1, self.slot_size)
        return torch.cdist(flat, self.codebook).argmin(dim=1).view(slots.shape[:-1])

    def guess_level1(self, words2):
        """(n, 4, 4) level-2 word numbers -> (n, 7, 7) best guess of the level-1 words."""
        return self.decoder(self.codebook[words2].permute(0, 3, 1, 2)).argmax(dim=1)


def train(words1: np.ndarray, level1_codebook: torch.Tensor, epochs: int = 6,
          batch_size: int = 128, seed: int = 42) -> WordVQVAE:
    """Learn the level-2 vocabulary from level-1 word grids alone (no labels)."""
    torch.manual_seed(seed)
    w_all = torch.tensor(words1, dtype=torch.long)
    model = WordVQVAE(level1_codebook)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs):
        order = torch.randperm(len(w_all))
        used = torch.zeros(len(model.codebook), dtype=torch.bool)
        total, right = 0.0, 0
        for start in range(0, len(w_all), batch_size):
            w = w_all[order[start:start + batch_size]]
            slots = model.slots(w)
            words2 = model.nearest_words(slots)
            snapped = model.codebook[words2]
            used[words2.flatten()] = True

            # Same three losses as level 1 (see fam/vqvae.py), but the rebuild
            # loss now scores the guessed level-1 words instead of pixels.
            passed_on = slots + (snapped - slots).detach()
            scores = model.decoder(passed_on.permute(0, 3, 1, 2))
            rebuild_loss = F.cross_entropy(scores, w)
            loss = (
                rebuild_loss
                + F.mse_loss(snapped, slots.detach())
                + 0.25 * F.mse_loss(slots, snapped.detach())
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += rebuild_loss.item() * len(w)
            right += (scores.argmax(dim=1) == w).sum().item()

        # Revive level-2 words nobody used this epoch, as at level 1.
        dead = (~used).nonzero().flatten()
        if len(dead) and epoch < epochs - 1:
            with torch.no_grad():
                sample = model.slots(w_all[torch.randint(len(w_all), (len(dead),))])
                model.codebook[dead] = sample[:, 1, 1]
        print(f"  epoch {epoch + 1}/{epochs}: rebuild loss {total / len(w_all):.3f}, "
              f"level-1 words recovered {right / w_all.numel():.0%}, "
              f"words used {int(used.sum())}/{len(model.codebook)}")
    return model


@torch.no_grad()
def words_of(model: WordVQVAE, words1: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Each 7x7 level-1 grid as a 4x4 grid of level-2 word numbers (uint8)."""
    out = [
        model.nearest_words(model.slots(torch.tensor(words1[i:i + batch_size], dtype=torch.long))).numpy()
        for i in range(0, len(words1), batch_size)
    ]
    return np.concatenate(out).astype(np.uint8)
