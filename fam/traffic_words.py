"""Hour words: write one sensor's last hour of traffic as 2 self-contained words.

    last hour (12 speeds) -> ENCODER -> 2 slots -> snap each to nearest WORD (256-word dictionary)
        -> each word draws its own layer -> layers ADDED -> rebuild the last hour AND predict the next hour

Same recipe as experiment 10 (fam/set_vqvae.py): the layers are added, so the
words form an unordered set and each must mean something on its own. Two
lessons from experiments 6-10 are built in:
  - words are trained to PREDICT (the next hour), not only to copy (the last
    hour), so a word can mean "slowing down, jam coming", not just "45 mph";
  - it's trained on the training months only, so it never sees the test future.
A sensor sends its neighbours 2 words = 2 bytes instead of 12 readings.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HOUR = 12  # readings per hour (one every 5 minutes)


class HourWords(nn.Module):
    def __init__(self, n_words: int = 2, slot_size: int = 8, vocab_size: int = 256):
        super().__init__()
        self.n_words, self.slot_size = n_words, slot_size
        self.encoder = nn.Sequential(
            nn.Linear(HOUR, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_words * slot_size),
        )
        # One shared drawer: ONE word -> its layer over the last hour + the next hour (24 numbers).
        self.drawer = nn.Sequential(nn.Linear(slot_size, 64), nn.ReLU(), nn.Linear(64, 2 * HOUR))
        self.codebook = nn.Parameter(torch.randn(vocab_size, slot_size) * 0.1)
        # Speeds are scaled to roughly -1..1 before the encoder (set from training data in train()).
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(1.0))

    def slots(self, hours):
        """(n, 12) speeds in mph -> (n, 2, slot_size) slot vectors."""
        return self.encoder((hours - self.mean) / self.scale).view(len(hours), self.n_words, self.slot_size)

    def nearest_words(self, slots):
        flat = slots.reshape(-1, self.slot_size)
        return torch.cdist(flat, self.codebook).argmin(dim=1).view(slots.shape[:-1])

    def draw(self, vectors):
        """(n, 2, slot_size) word vectors -> (n, 24) scaled speeds: last hour then next hour."""
        n = len(vectors)
        return self.drawer(vectors.reshape(-1, self.slot_size)).view(n, self.n_words, 2 * HOUR).sum(dim=1)


def train(past: np.ndarray, future: np.ndarray, epochs: int = 5, batch_size: int = 1024,
          seed: int = 42) -> HourWords:
    """Learn the dictionary from (last hour, next hour) pairs of the training months. No labels."""
    torch.manual_seed(seed)
    model = HourWords()
    model.mean.fill_(float(past.mean()))
    model.scale.fill_(float(past.std()))
    x_all = torch.tensor(past, dtype=torch.float32)
    target_all = (torch.tensor(np.concatenate([past, future], axis=1), dtype=torch.float32) - model.mean) / model.scale
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for epoch in range(epochs):
        order = torch.randperm(len(x_all))
        used = torch.zeros(len(model.codebook), dtype=torch.bool)
        total_past = total_next = 0.0
        for start in range(0, len(x_all), batch_size):
            batch = order[start:start + batch_size]
            slots = model.slots(x_all[batch])
            words = model.nearest_words(slots)
            snapped = model.codebook[words]
            used[words.flatten()] = True

            # Same three losses as the other VQ-VAEs (see fam/vqvae.py); the rebuild
            # loss covers both the last hour (describe) and the next hour (predict).
            passed_on = slots + (snapped - slots).detach()
            drawn, target = model.draw(passed_on), target_all[batch]
            loss_past = F.mse_loss(drawn[:, :HOUR], target[:, :HOUR])
            loss_next = F.mse_loss(drawn[:, HOUR:], target[:, HOUR:])
            loss = (
                loss_past + loss_next
                + F.mse_loss(snapped, slots.detach())
                + 0.25 * F.mse_loss(slots, snapped.detach())
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_past += loss_past.item() * len(batch)
            total_next += loss_next.item() * len(batch)

        # Revive words nobody used this epoch by moving them onto real slots.
        dead = (~used).nonzero().flatten()
        if len(dead) and epoch < epochs - 1:
            with torch.no_grad():
                sample = model.slots(x_all[torch.randint(len(x_all), (len(dead),))])
                model.codebook[dead] = sample[torch.arange(len(dead)), torch.randint(model.n_words, (len(dead),))]
        # Errors back in mph: sqrt(mean squared scaled error) * scale.
        to_mph = float(model.scale)
        print(f"  epoch {epoch + 1}/{epochs}: error last hour {np.sqrt(total_past / len(x_all)) * to_mph:.2f} mph, "
              f"next hour {np.sqrt(total_next / len(x_all)) * to_mph:.2f} mph, words used {int(used.sum())}/256")
    return model


@torch.no_grad()
def words_of(model: HourWords, hours: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    """(n, 12) hours of speeds -> (n, 2) word numbers (uint8, one byte each)."""
    out = [
        model.nearest_words(model.slots(torch.tensor(hours[i:i + batch_size], dtype=torch.float32))).numpy()
        for i in range(0, len(hours), batch_size)
    ]
    return np.concatenate(out).astype(np.uint8)
