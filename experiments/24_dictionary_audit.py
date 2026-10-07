"""Experiment 24: a dictionary audit. Is something specific wrong with experiment 22b's dictionary?

Experiment 23 failed on a healthy dictionary. Before choosing a new direction, three checks
on that same dictionary (nothing new built), on the training weeks only: fit on week 1
(Mon 4 - Sun 10 Nov), check on week 2 (Mon 11 - Sun 17 Nov). The test week isn't touched.

    A  information actually used: per word position, the entropy of its word use (bits, max 8)
       specific cause if the 16 positions together carry under 64 bits (half of 128)
    B  rebuild error by hour: the 16-word rebuild error for each of the 24 hours in the window
       specific cause if the last hour's error is at least 1.25x the 24-hour average
    C  before vs after snapping: next-hour internet and calls forecast from the encoder's 32
       numbers, from the 16-word message, and from raw data (LightGBM, experiment 20's settings)
       specific cause if words >= 1.2x encoder numbers (snapping loses it),
       or encoder numbers >= 1.5x raw data (the encoder's squeeze loses it)
Decision rule committed in the README before running.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import lightgbm as lgb
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from fam import residual_words, telecom

HISTORY, WEEK = 24, 168
SETTINGS = dict(n_estimators=300, learning_rate=0.1, num_leaves=63, random_state=0, n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

logs = np.log1p(telecom.load_hourly())[:, :2 * WEEK]                 # the two training weeks only
n_sq = len(logs)
mean, spread = logs.mean(axis=(0, 1)), logs.std(axis=(0, 1))          # the dictionary's own statistics
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
fit_t = np.arange(HISTORY - 1, WEEK - 1)                               # t + 1 in week 1
check_t = np.arange(WEEK - 1, 2 * WEEK - 1)                            # t + 1 in week 2

model = residual_words.CorrectionWords()
model.load_state_dict(torch.load(ROOT / "data" / "models" / "telecom_correction_words_exp22b.pt"))
model.eval()
books = model.codebooks.detach().numpy()


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


@torch.no_grad()
def encode(t_values):
    """Encoder numbers (before snapping) and the 16 words, a day of hours at a time."""
    zs, ws = [], []
    for i in range(0, len(t_values), 24):
        x = torch.tensor(standardise(history(t_values[i:i + 24])))
        z = model.encoder(x)
        zs.append(z.numpy().reshape(n_sq, -1, 32))
        ws.append(model.quantize(z)[0].numpy().astype(np.uint8).reshape(n_sq, -1, 16))
    return np.concatenate(zs, axis=1).reshape(-1, 32), np.concatenate(ws, axis=1).reshape(-1, 16)


z_fit, w_fit = encode(fit_t)
z_check, w_check = encode(check_t)
print(f"Encoded {len(z_fit):,} week-1 and {len(z_check):,} week-2 histories ({time.time() - start_all:.0f} s)")

# A. Information actually used, per word position.
bits = []
for k in range(16):
    p = np.bincount(w_check[:, k], minlength=256) / len(w_check)
    p = p[p > 0]
    bits.append(float(-(p * np.log2(p)).sum()))
total_bits = sum(bits)
print("\nA. Information per word position (bits, max 8): " + " ".join(f"{b:.1f}" for b in bits))
print(f"   Together: {total_bits:.1f} of 128 bits ({total_bits / 8:.1f} of 16 bytes)  "
      f"-> {'SPECIFIC CAUSE: collapse wastes the message' if total_bits < 64 else 'no specific cause'}")

# B. Rebuild error by hour, 16 words, on a sample of week-2 histories.
rng = np.random.default_rng(0)
sample = rng.choice(len(w_check), size=200_000, replace=False)
hist_check = history(check_t)
x_sample = standardise(hist_check[sample])
del hist_check
prefix16 = sum(books[j][w_check[sample, j]] for j in range(16)).astype(np.float32)
with torch.no_grad():
    rebuilt = model.decoder(torch.tensor(prefix16)).numpy()
per_hour = ((rebuilt - x_sample) ** 2).reshape(-1, HISTORY, 3).mean(axis=(0, 2))
ratio = per_hour[-1] / per_hour.mean()
print("\nB. 16-word rebuild error by hour in the window (oldest -> most recent):")
print("   " + " ".join(f"{e:.3f}" for e in per_hour))
print(f"   Last hour {per_hour[-1]:.3f} vs 24-hour average {per_hour.mean():.3f}: {ratio:.2f}x  "
      f"-> {'SPECIFIC CAUSE: the training goal does not fit forecasting' if ratio >= 1.25 else 'no specific cause'}")


# C. Before vs after snapping: forecasting on week 2 from models fitted on week 1.
def extras(t_values):
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


def prefix_all(words):
    return sum(books[j][words[:, j]] for j in range(16)).astype(np.float32)


targets = {"internet": 2, "calls": 1}
inputs = {
    "raw data": (np.hstack([history(fit_t).astype(np.float32), extras(fit_t)]),
                 np.hstack([history(check_t).astype(np.float32), extras(check_t)])),
    "encoder numbers": (np.hstack([z_fit, extras(fit_t)]), np.hstack([z_check, extras(check_t)])),
    "16 words": (np.hstack([prefix_all(w_fit), extras(fit_t)]), np.hstack([prefix_all(w_check), extras(check_t)])),
}
err = {}
for name, (X_fit, X_check) in inputs.items():
    for q, a in targets.items():
        y_fit, y_check = logs[:, fit_t + 1, a].reshape(-1), logs[:, check_t + 1, a].reshape(-1)
        fitted = lgb.LGBMRegressor(**SETTINGS).fit(X_fit, y_fit)
        err[name, q] = float(np.abs(fitted.predict(X_check) - y_check).mean())
    print(f"  {name} done ({time.time() - start_all:.0f} s)")

print("\nC. Next-hour error on week 2 (mean absolute error of log(1 + x)):")
print(f"   {'':<17}{'internet':>10}{'calls':>10}")
for name in inputs:
    print(f"   {name:<17}{err[name, 'internet']:>10.4f}{err[name, 'calls']:>10.4f}")
causes = []
for q in targets:
    snap = err["16 words", q] / err["encoder numbers", q]
    squeeze = err["encoder numbers", q] / err["raw data", q]
    print(f"   {q}: words / encoder numbers {snap:.2f}x (cause if >= 1.2); encoder numbers / raw {squeeze:.2f}x (cause if >= 1.5)")
    if snap >= 1.2:
        causes.append(f"snapping to words loses it ({q})")
    if squeeze >= 1.5:
        causes.append(f"the encoder's squeeze loses it ({q})")
if total_bits < 64:
    causes.insert(0, "collapse wastes the message")
if ratio >= 1.25:
    causes.insert(0, "the training goal does not fit forecasting")
print("\nAudit verdict: " + ("SPECIFIC CAUSES: " + "; ".join(causes) if causes else
                             "NO SPECIFIC CAUSE: it's the idea in this form, not the engineering"))
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
