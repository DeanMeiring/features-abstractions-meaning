"""Experiment 21a: why did experiment 21's correction-word training break? Training data only.

Run 1 of experiment 21: rebuild error with 16 words was 0.097 after the first pass,
then rose every pass to 2.997. Two suspects: the revive step (moving unused entries
onto real data shifts what every later dictionary describes) and the learning rate.
Four versions are trained on the same sample of training-week windows and checked
after every pass on other training-week windows. The test week isn't touched, and
no forecasting is done: this only finds the version whose training doesn't break.

    A  revive, lr 0.002     (as run 1)
    B  no revive, lr 0.002
    C  revive, lr 0.0005
    D  no revive, lr 0.0005
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
VERSIONS = {"A revive, lr 0.002": (True, 2e-3), "B no revive, lr 0.002": (False, 2e-3),
            "C revive, lr 0.0005": (True, 5e-4), "D no revive, lr 0.0005": (False, 5e-4)}

logs = np.log1p(telecom.load_hourly())[:, :telecom.TRAIN_HOURS]      # training weeks only
mean, spread = logs.mean(axis=(0, 1)), logs.std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2).reshape(-1, HISTORY * 3)
rng = np.random.default_rng(0)
pick = rng.choice(len(windows), size=320_000, replace=False)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


fit, check = standardise(windows[pick[:300_000]]), standardise(windows[pick[300_000:]])
print(f"Fit on {len(fit):,} training-week windows, check on {len(check):,} others\n")

final = {}
for name, (revive, lr) in VERSIONS.items():
    print(name)
    t = time.time()
    model = residual_words.train(fit, epochs=6, revive=revive, lr=lr, check=check)
    final[name] = residual_words.rebuild_errors(model, check)
    print(f"  took {time.time() - t:.0f} s\n")

print("After 6 passes, rebuild error on the check windows (1 / 4 / 16 words), and whether 16 words beat 1:")
for name, e in final.items():
    print(f"  {name:<24} {e[0]:.3f} / {e[3]:.3f} / {e[15]:.3f}   {'improves with words' if e[15] < e[0] else 'BROKEN'}")
