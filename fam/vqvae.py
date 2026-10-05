"""A VQ-VAE: like the autoencoder, but the middle is made of words from a vocabulary.

    customer row -> ENCODER -> 4 slots -> snap each slot to the nearest WORD -> DECODER -> rebuilt row

The vocabulary (the "codebook") is a list of 256 learned vectors. Each slot
the encoder produces is replaced by the closest vocabulary vector, so a
customer becomes 4 word numbers, e.g. [17, 203, 17, 88]. Each word number
fits in one byte (0-255), so a customer is stored in 4 bytes.

Whoever has the codebook can turn the 4 word numbers back into vectors: the
codebook is the shared dictionary, the 4 numbers are the message.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fam.autoencoder import to_numbers


class VQVAE(nn.Module):
    def __init__(self, n_inputs: int, n_slots: int, slot_size: int, vocab_size: int):
        super().__init__()
        self.n_slots, self.slot_size = n_slots, slot_size
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 32), nn.ReLU(), nn.Linear(32, n_slots * slot_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_slots * slot_size, 32), nn.ReLU(), nn.Linear(32, n_inputs)
        )
        # The vocabulary: vocab_size learned vectors, shared by all slots.
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size))

    def slots(self, x):
        """Encoder output, cut into n_slots vectors per customer."""
        return self.encoder(x).view(-1, self.n_slots, self.slot_size)

    def nearest_words(self, slots):
        """For each slot, the number of the closest vocabulary vector."""
        distances = torch.cdist(slots.reshape(-1, self.slot_size), self.codebook)
        return distances.argmin(dim=1).view(-1, self.n_slots)


class CustomerWordEncoder:
    """Learns a vocabulary from training rows, then writes any customer as words."""

    def __init__(self, n_slots=4, slot_size=4, vocab_size=256, epochs=600, seed=42):
        self.n_slots, self.slot_size, self.vocab_size = n_slots, slot_size, vocab_size
        self.epochs, self.seed = epochs, seed

    def fit(self, X: pd.DataFrame) -> "CustomerWordEncoder":
        torch.manual_seed(self.seed)
        nums = to_numbers(X)
        self.columns = nums.columns
        self.mean, self.std = nums.mean(), nums.std().replace(0, 1)
        x = self._scale(nums)

        self.model = VQVAE(x.shape[1], self.n_slots, self.slot_size, self.vocab_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            slots = self.model.slots(x)
            words = self.model.nearest_words(slots)
            snapped = self.model.codebook[words]  # each slot replaced by its word's vector

            # Trick ("straight-through"): snapping to the nearest word has no
            # slope, so the encoder couldn't learn through it. We pretend the
            # snap didn't happen when working out the encoder's updates.
            passed_on = slots + (snapped - slots).detach()
            rebuilt = self.model.decoder(passed_on.reshape(len(x), -1))

            rebuild_loss = F.mse_loss(rebuilt, x)
            # Pull the chosen words toward what the encoder produced...
            vocab_loss = F.mse_loss(snapped, slots.detach())
            # ...and keep the encoder close to the words, so it doesn't drift.
            commit_loss = F.mse_loss(slots, snapped.detach())
            loss = rebuild_loss + vocab_loss + 0.25 * commit_loss
            loss.backward()
            optimizer.step()

            # Words nobody uses never get updated ("dead words"). Every 50
            # rounds, move them onto random slots so they get another chance.
            if epoch % 50 == 0 and epoch < self.epochs - 100:
                self._revive_dead_words(slots.detach(), words)

        self.final_loss = rebuild_loss.item()
        self.words_used = int(self.words(X).stack().nunique())
        return self

    def words(self, X: pd.DataFrame) -> pd.DataFrame:
        """Each customer as n_slots word numbers (0 to vocab_size - 1)."""
        with torch.no_grad():
            w = self.model.nearest_words(self.model.slots(self._prepare(X))).numpy()
        names = [f"word_{i}" for i in range(self.n_slots)]
        return pd.DataFrame(w.astype(np.uint8), columns=names, index=X.index)

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Look the words up in the codebook: the vectors a model reads."""
        w = torch.tensor(self.words(X).to_numpy(dtype=np.int64))
        with torch.no_grad():
            vectors = self.model.codebook[w].reshape(len(X), -1).numpy()
        names = [f"vq_{i}" for i in range(vectors.shape[1])]
        return pd.DataFrame(vectors, columns=names, index=X.index)

    def _revive_dead_words(self, slots, words):
        used = torch.zeros(self.vocab_size, dtype=torch.bool)
        used[words.flatten()] = True
        dead = (~used).nonzero().flatten()
        if len(dead):
            all_slots = slots.reshape(-1, self.slot_size)
            pick = torch.randint(len(all_slots), (len(dead),))
            with torch.no_grad():
                self.model.codebook[dead] = all_slots[pick]

    def _prepare(self, X):
        return self._scale(to_numbers(X).reindex(columns=self.columns, fill_value=0.0))

    def _scale(self, nums):
        return torch.tensor(((nums - self.mean) / self.std).to_numpy(dtype=np.float32))
