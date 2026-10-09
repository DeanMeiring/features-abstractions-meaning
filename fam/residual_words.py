"""Correction words: write a 24-hour history as up to 16 words, coarse to fine.

    history (72 numbers) -> ENCODER -> one 32-number vector z
    word 1  = nearest entry of dictionary 1 to z                      ("a normal weekday-evening square")
    word k  = nearest entry of dictionary k to what's still missing    ("but 40% busier right now")
              (z minus the sum of words 1 .. k-1)
    the sum of the first k words -> DECODER -> rebuilt history

The idea comes from predictive coding: the first word is a prediction, and each
further word carries only the error that remains, so a message can stop early
when an item is unsurprising. In ML terms this is residual quantization. The
decoder is trained on every message length k = 1 .. 16 at once, so every shorter
message is usable. Trained only to rebuild the history (no future hours).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CorrectionWords(nn.Module):
    def __init__(self, size: int = 72, latent: int = 32, n_words: int = 16, vocab_size: int = 256, future: int = 0):
        """future (experiment 25): how many future numbers the decoder also predicts (0 = rebuild only)."""
        super().__init__()
        self.size, self.latent, self.n_words, self.future = size, latent, n_words, future
        self.encoder = nn.Sequential(nn.Linear(size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, size + future))
        # One 256-word dictionary per position: dictionary k only ever describes what words 1..k-1 missed.
        self.codebooks = nn.Parameter(torch.randn(n_words, vocab_size, latent) * 0.1)

    def quantize(self, z):
        """z (n, latent) -> word numbers (n, 16), each word's vector (n, 16, latent), what each word saw (n, 16, latent)."""
        residual, ids, vectors, seen = z, [], [], []
        for k in range(self.n_words):
            idx = torch.cdist(residual, self.codebooks[k]).argmin(dim=1)
            q = self.codebooks[k][idx]
            ids.append(idx)
            vectors.append(q)
            seen.append(residual)
            residual = residual - q
        return torch.stack(ids, 1), torch.stack(vectors, 1), torch.stack(seen, 1)

    def prefix(self, ids, k):
        """Sum of the first k words' vectors: (n, k_or_more) word numbers -> (n, latent)."""
        return sum(self.codebooks[j][ids[:, j]] for j in range(k))


def train(windows: np.ndarray, epochs: int = 6, batch_size: int = 1024, seed: int = 0, lr: float = 2e-3,
          revive: bool = True, check: np.ndarray | None = None, lr_to_zero: bool = False,
          keep_best: bool = False, future: np.ndarray | None = None,
          check_future: np.ndarray | None = None, patience: int | None = None) -> CorrectionWords:
    """Learn the 16 dictionaries from standardised training windows alone (no targets).

    lr, revive: the two suspects in experiment 21 run 1's breakdown (experiment 21a).
    check: other windows to measure the rebuild error on after every pass (eval mode).
    lr_to_zero (experiment 22): lower the learning rate linearly to zero over all passes,
        so the steps get smaller as training settles.
    keep_best (experiment 22): return the pass with the lowest 16-word rebuild error on
        `check`, not the last pass. The model's `history` attribute keeps every pass's
        check errors (1..16 words), for the health gate.
    future, check_future (experiment 25): the next hours after each window. The message must
        then also predict them; rebuild and prediction errors are weighted equally, and the
        check errors (history, keep_best) are their sum.
    patience (experiment 29): stop early when the 16-word check error hasn't improved for
        this many passes (needs `check`); with keep_best, the best pass is returned.
    """
    torch.manual_seed(seed)
    x_all = torch.tensor(windows, dtype=torch.float32)
    f_all = None if future is None else torch.tensor(future, dtype=torch.float32)
    model = CorrectionWords(size=windows.shape[1], future=0 if future is None else future.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * -(-len(x_all) // batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / steps if lr_to_zero else 1.0)
    history, best, best_state, best_epoch = [], np.inf, None, 0
    for epoch in range(epochs):
        order = torch.randperm(len(x_all))
        used = torch.zeros(model.n_words, model.codebooks.shape[1], dtype=torch.bool)
        totals = np.zeros(model.n_words)
        for start in range(0, len(x_all), batch_size):
            batch = order[start:start + batch_size]
            x = x_all[batch]
            z = model.encoder(x)
            ids, vectors, seen = model.quantize(z)
            for k in range(model.n_words):
                used[k, ids[:, k]] = True
            # Rebuild from every message length at once, so short messages work too.
            # The straight-through trick (see fam/vqvae.py) lets learning reach the encoder.
            prefixes = torch.cumsum(vectors, dim=1)                      # (n, 16, latent)
            passed_on = z[:, None] + (prefixes - z[:, None]).detach()
            rebuilt = model.decoder(passed_on)                           # (n, 16, 72 [+ future])
            per_k = ((rebuilt[..., :model.size] - x[:, None]) ** 2).mean(dim=(0, 2))   # rebuild error for each k
            if f_all is not None:                                        # + prediction error for each k
                per_k = per_k + ((rebuilt[..., model.size:] - f_all[batch][:, None]) ** 2).mean(dim=(0, 2))
            loss = (per_k.mean()
                    + F.mse_loss(vectors, seen.detach())                 # each word moves toward what it described
                    + 0.25 * F.mse_loss(z, prefixes[:, -1].detach()))    # the encoder commits to its message
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            totals += per_k.detach().numpy() * len(x)
        # Revive entries nobody used, on residuals that dictionary actually sees.
        if revive and epoch < epochs - 1:
            with torch.no_grad():
                sample = x_all[torch.randint(len(x_all), (4096,))]
                _, _, seen = model.quantize(model.encoder(sample))
                for k in range(model.n_words):
                    dead = (~used[k]).nonzero().flatten()
                    if len(dead):
                        model.codebooks[k, dead] = seen[torch.randint(len(sample), (len(dead),)), k]
        errors = totals / len(x_all) if check is None else rebuild_errors(model, check, check_future)
        history.append(errors)
        if errors[15] < best:
            best, best_epoch = errors[15], epoch
            if keep_best:
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch + 1}/{epochs}: rebuild error with 1 / 4 / 16 words {errors[0]:.3f} / {errors[3]:.3f} / "
              f"{errors[15]:.3f}, entries used {int(used.sum())}/{used.numel()}", flush=True)
        model.train()
        if patience is not None and epoch - best_epoch >= patience:
            print(f"  stopped early: no improvement for {patience} passes (best was pass {best_epoch + 1})", flush=True)
            break
    if keep_best:
        model.load_state_dict(best_state)
    model.history = np.array(history)
    return model.eval()


@torch.no_grad()
def rebuild_errors(model: CorrectionWords, windows: np.ndarray, future: np.ndarray | None = None) -> np.ndarray:
    """Rebuild error for each message length k = 1..16 on the given windows (past part only),
    plus the prediction error of the next hours when `future` is given."""
    model.eval()
    x = torch.tensor(windows, dtype=torch.float32)
    _, vectors, _ = model.quantize(model.encoder(x))
    rebuilt = model.decoder(torch.cumsum(vectors, dim=1))
    errors = ((rebuilt[..., :model.size] - x[:, None]) ** 2).mean(dim=(0, 2))
    if future is not None:
        errors = errors + ((rebuilt[..., model.size:] - torch.tensor(future, dtype=torch.float32)[:, None]) ** 2).mean(dim=(0, 2))
    return errors.numpy()


@torch.no_grad()
def words_of(model: CorrectionWords, windows: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    """(n, 72) standardised histories -> (n, 16) word numbers (uint8), coarse to fine."""
    out = [model.quantize(model.encoder(torch.tensor(windows[i:i + batch_size], dtype=torch.float32)))[0].numpy()
           for i in range(0, len(windows), batch_size)]
    return np.concatenate(out).astype(np.uint8)
