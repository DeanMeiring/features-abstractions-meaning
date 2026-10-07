"""Experiment 22a: is the revive step what breaks the correction-word training? Training data only.

The engineering review after experiment 22 named revive as the prime suspect: after every
pass it jumps unused entries to new positions in all 16 dictionaries, including the first,
whose leftovers every later dictionary describes. A jump isn't scaled by the learning rate,
which fits experiment 22 breaking while the learning rate shrank toward zero.

This is experiment 22 exactly (same 1 million windows, the same check windows, learning
rate 0.0005 lowered to zero, best pass kept), with one change: revive switched off.
No forecasting, the test week isn't touched, and nothing is built from it.

Decision rule, written before running:
    stable  (last pass within 10% of the best)  -> revive is confirmed as the trigger
    breaks  (last pass more than 10% above)     -> revive isn't the whole cause; the chain
                                                   of hard choices and the straight-through
                                                   mismatch remain the suspects
Either way, it only informs the design of the next experiment (the anchored training),
which gets its own bar before it's built.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fam import residual_words, telecom

HISTORY = 24
start_all = time.time()

# Exactly experiment 22's windows.
logs = np.log1p(telecom.load_hourly())
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))
spread = logs[:, :train_hours].std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)
all_train = windows[:, train_t - (HISTORY - 1)].reshape(-1, HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


fit_rows = np.random.default_rng(0).choice(len(all_train), size=1_000_000, replace=False)
others = np.setdiff1d(np.arange(len(all_train)), fit_rows)
check_rows = np.random.default_rng(1).choice(others, size=20_000, replace=False)
fit, check = standardise(all_train[fit_rows]), standardise(all_train[check_rows])
del all_train
print(f"Training on {len(fit):,} windows, checking on {len(check):,} others (training weeks only), revive OFF")

model = residual_words.train(fit, epochs=6, lr=5e-4, check=check, lr_to_zero=True, keep_best=True, revive=False)
history = model.history
best = int(history[:, 15].argmin())
stable = history[-1, 15] <= 1.10 * history[best, 15]
print(f"\nRebuild error with 16 words, pass by pass: " + " / ".join(f"{e:.3f}" for e in history[:, 15]))
print(f"Rebuild error with 1 word, pass by pass:   " + " / ".join(f"{e:.3f}" for e in history[:, 0]))
print(f"Best pass {best + 1}: {history[best, 15]:.3f}; last pass {history[-1, 15]:.3f}")
print("-> " + ("STABLE: revive is confirmed as the trigger" if stable else
               "BREAKS: revive isn't the whole cause; the chain and the straight-through mismatch remain suspects"))
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
