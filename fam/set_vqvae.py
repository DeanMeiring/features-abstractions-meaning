"""Set VQ-VAE: write an image as an unordered SET of 8 words that each stand on their own.

    image -> ENCODER (sees the whole image) -> 8 slots -> snap each to nearest WORD
          -> each word draws its own picture layer -> layers are ADDED -> rebuilt image

Grid words (fam/image_vqvae.py) only mean something at their grid position:
"an edge, here". Real words mean the same thing wherever they appear. Here,
the picture layers are added together, and adding doesn't care about order,
so neither slot number nor position can carry meaning. Each word has to say
everything about its own part of the image (what, and where), like
"boot sole" or "dark top half". An image is 8 words = 8 bytes.

alphabet=True (experiment 14) builds each symbol from two parts, like a
Chinese character from a radical and another component: each slot snaps to
the nearest of 16 RADICALS, and whatever the radical doesn't explain snaps to
the nearest of 16 DETAILS. Symbol = radical + detail: still 16 x 16 = 256
symbols, 1 byte each, but symbols sharing a radical are related.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fam.image_vqvae import to_tensor


class SetVQVAE(nn.Module):
    def __init__(self, n_words: int = 8, slot_size: int = 16, vocab_size: int = 256, alphabet: bool = False):
        super().__init__()
        self.n_words, self.slot_size, self.alphabet = n_words, slot_size, alphabet
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
        if alphabet:
            side = int(vocab_size ** 0.5)  # 16 radicals x 16 details = 256 symbols
            self.radicals = nn.Parameter(torch.randn(side, slot_size) * 0.1)
            self.details = nn.Parameter(torch.randn(side, slot_size) * 0.02)
        else:
            self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)

    def vectors(self):
        """The dictionary: one vector per symbol, (256, slot_size)."""
        if not self.alphabet:
            return self.codebook
        return (self.radicals[:, None] + self.details[None, :]).reshape(-1, self.slot_size)

    def quantize(self, slots):
        """Snap slots to symbols: (symbol numbers, their vectors). See the module docstring."""
        flat = slots.reshape(-1, self.slot_size)
        if not self.alphabet:
            ids = torch.cdist(flat, self.codebook).argmin(dim=1)
            snapped = self.codebook[ids]
        else:
            radical = torch.cdist(flat, self.radicals).argmin(dim=1)
            leftover = flat - self.radicals[radical]
            detail = torch.cdist(leftover, self.details).argmin(dim=1)
            ids = radical * len(self.details) + detail
            snapped = self.radicals[radical] + self.details[detail]
        return ids.view(slots.shape[:-1]), snapped.view(slots.shape)

    def slots(self, x):
        """(n, 1, 28, 28) images -> (n, 8, slot_size) slot vectors."""
        return self.encoder(x).view(len(x), self.n_words, self.slot_size)

    def nearest_words(self, slots):
        return self.quantize(slots)[0]

    def layers(self, vectors):
        """(n, 8, slot_size) word vectors -> (n, 8, 28, 28) one picture layer per word."""
        n = len(vectors)
        return self.drawer(vectors.reshape(-1, self.slot_size)).view(n, self.n_words, 28, 28)

    def draw(self, vectors):
        """Add the layers (order can't matter) and squash to brightness 0-1."""
        return torch.sigmoid(self.layers(vectors).sum(dim=1, keepdim=True))

    def decode_words(self, words):
        """(n, 8) word numbers -> (n, 1, 28, 28) rebuilt images."""
        return self.draw(self.vectors()[words])


def train(images: np.ndarray, epochs: int = 6, batch_size: int = 128, seed: int = 42,
          alphabet: bool = False, dictionary: SetVQVAE | None = None) -> SetVQVAE:
    """Learn the vocabulary from images alone (no labels).

    dictionary (experiment 15): instead of inventing a language, a NEW SPEAKER
    learns an existing one. The given model's symbols and drawer are copied and
    frozen; only a fresh encoder learns, and only from being understood (the
    frozen drawer must rebuild the image from its symbols).
    """
    torch.manual_seed(seed)
    x_all = to_tensor(images)
    if dictionary is None:
        model = SetVQVAE(alphabet=alphabet)
        learning = list(model.parameters())
    else:
        alphabet = dictionary.alphabet
        model = SetVQVAE(alphabet=alphabet)  # fresh, randomly started encoder
        frozen = {k: v for k, v in dictionary.state_dict().items() if not k.startswith("encoder.")}
        model.load_state_dict(frozen, strict=False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("encoder.")
        learning = list(model.encoder.parameters())
    optimizer = torch.optim.Adam(learning, lr=2e-3)

    for epoch in range(epochs):
        order = torch.randperm(len(x_all))
        used = torch.zeros(256, dtype=torch.bool)
        total = 0.0
        for start in range(0, len(x_all), batch_size):
            x = x_all[order[start:start + batch_size]]
            slots = model.slots(x)
            words, snapped = model.quantize(slots)
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

        if epoch < epochs - 1 and dictionary is None:  # a frozen dictionary never changes
            revive(model, x_all, used)
        radicals = f", radicals used {int(used.view(16, 16).any(dim=1).sum())}/16" if alphabet else ""
        print(f"  epoch {epoch + 1}/{epochs}: rebuild error {total / len(x_all):.4f}, "
              f"symbols used {int(used.sum())}/256{radicals}")
    return model


@torch.no_grad()
def revive(model: SetVQVAE, x_all: torch.Tensor, used: torch.Tensor) -> None:
    """Move symbols (or radicals / details) nobody used this epoch onto real slots."""
    if not model.alphabet:
        dead = (~used).nonzero().flatten()
        if len(dead):
            sample = model.slots(x_all[torch.randint(len(x_all), (len(dead),))])
            model.codebook[dead] = sample[torch.arange(len(dead)), torch.randint(model.n_words, (len(dead),))]
        return
    grid = used.view(16, 16)
    dead_radicals = (~grid.any(dim=1)).nonzero().flatten()
    dead_details = (~grid.any(dim=0)).nonzero().flatten()
    if len(dead_radicals) + len(dead_details) == 0:
        return
    sample = model.slots(x_all[torch.randint(len(x_all), (32,))])
    sample = sample[torch.arange(32), torch.randint(model.n_words, (32,))]  # one random slot per image
    model.radicals[dead_radicals] = sample[:len(dead_radicals)]
    leftover = sample - model.radicals[torch.cdist(sample, model.radicals).argmin(dim=1)]
    model.details[dead_details] = leftover[16:16 + len(dead_details)]


@torch.no_grad()
def words_of(model: SetVQVAE, images: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    """Each image as 8 word numbers (uint8, one byte each)."""
    out = [
        model.nearest_words(model.slots(to_tensor(images[i:i + batch_size]))).numpy()
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(out).astype(np.uint8)
