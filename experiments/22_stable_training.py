"""Experiment 22: can the correction-word dictionary be trained properly? Training data only.

Experiment 21's coarse-to-fine dictionary broke in both runs: the rebuild error fell,
then rose until the dictionary carried almost nothing. Here the same dictionary and the
same 1 million training windows, with two fixes fixed in advance:
    - learning rate 0.0005, lowered linearly to zero over the 6 passes
    - keep the best pass (lowest 16-word rebuild error on check windows), not the last
No forecasting, and the test week isn't touched.

Bar (README, committed before running):
    healthy if (1) the last pass's 16-word rebuild error is within 10% of the best pass's,
    (2) the kept dictionary's 16-word rebuild error is at most 0.10, and
    (3) the error falls from 1 to 4 to 16 words; otherwise fail.
Stop rule: if this fails, nothing new is built; back to the engineering problem first.
    python experiments/22_stable_training.py --smoke   # all 6 passes on a small sample (not a result)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from fam import residual_words, telecom

SMOKE = "--smoke" in sys.argv
MODEL_FILE = ROOT / "data" / "models" / ("telecom_correction_words_exp22_smoke.pt" if SMOKE else "telecom_correction_words_exp22.pt")
HISTORY = 24
start_all = time.time()

# Experiment 21's training windows: every square's 24-hour histories in the training weeks.
logs = np.log1p(telecom.load_hourly())
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))
spread = logs[:, :train_hours].std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)
all_train = windows[:, train_t - (HISTORY - 1)].reshape(-1, HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


# The same 1 million windows as experiment 21 (seed 0), and 20,000 other training windows to check on.
fit_rows = np.random.default_rng(0).choice(len(all_train), size=1_000_000, replace=False)
others = np.setdiff1d(np.arange(len(all_train)), fit_rows)
check_rows = np.random.default_rng(1).choice(others, size=20_000, replace=False)
if SMOKE:
    fit_rows = fit_rows[:50_000]
fit, check = standardise(all_train[fit_rows]), standardise(all_train[check_rows])
del all_train
print(f"Training on {len(fit):,} windows, checking on {len(check):,} others (training weeks only)")

t = time.time()
model = residual_words.train(fit, epochs=6, lr=5e-4, check=check, lr_to_zero=True, keep_best=True)
print(f"  took {time.time() - t:.0f} s")
torch.save(model.state_dict(), MODEL_FILE)

# The health gate.
history = model.history                         # (passes, 16): check errors per message length
best_pass = int(history[:, 15].argmin())
kept = residual_words.rebuild_errors(model, check)
not_broken = history[-1, 15] <= 1.10 * history[best_pass, 15]
good_enough = kept[15] <= 0.10
words_help = kept[0] > kept[3] > kept[15]

print(f"\nRebuild error with 16 words, pass by pass: " + " / ".join(f"{e:.3f}" for e in history[:, 15]))
print(f"Kept: pass {best_pass + 1}; rebuild error with 1 / 4 / 16 words {kept[0]:.3f} / {kept[3]:.3f} / {kept[15]:.3f}")
print("\nHealth gate:")
print(f"  (1) last pass within 10% of the best: {history[-1, 15]:.3f} vs {history[best_pass, 15]:.3f}  "
      f"{'yes' if not_broken else 'NO'}")
print(f"  (2) kept 16-word rebuild error at most 0.10: {kept[15]:.3f}  {'yes' if good_enough else 'NO'}")
print(f"  (3) more words help (1 > 4 > 16 words): {'yes' if words_help else 'NO'}")
healthy = not_broken and good_enough and words_help
print("  -> " + ("HEALTHY: experiment 23 may run on this dictionary" if healthy else
                 "FAIL: no forecasting; back to the engineering problem first (stop rule)"))
print(f"Saved {MODEL_FILE.name}. Total time: {(time.time() - start_all) / 60:.1f} min"
      + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
