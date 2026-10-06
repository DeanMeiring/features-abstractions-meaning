"""Experiment 20a: apply experiment 20's share rule. Raw data only, training weeks only.

The rule (README, committed before this runs): use the largest share from
{10%, 5%, 2%, 1%, 0.5%, 0.2%, 0.1%} at which raw data trained on that share is
worse than raw data trained on 100% by at least 5% on at least 2 of the 3
questions. Measured inside the training weeks: fit on week 1 (Mon 4 - Sun 10 Nov),
check on week 2 (Mon 11 - Sun 17 Nov). The test week isn't touched, and the words
aren't involved at all. Same inputs, questions and model settings as experiment 20,
except that "typical" for the unusual flag comes from week 1 only here.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import lightgbm as lgb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fam import telecom

SHARES = [0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
HISTORY, WEEK = 24, 168
SETTINGS = dict(n_estimators=300, learning_rate=0.1, num_leaves=63, random_state=0, n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

logs = np.log1p(telecom.load_hourly()[:, :2 * WEEK])           # the two training weeks only
n_sq = len(logs)
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
fit_t = np.arange(HISTORY - 1, WEEK - 1)                       # t + 1 in week 1
check_t = np.arange(WEEK - 1, 2 * WEEK - 1)                    # t + 1 in week 2


def inputs(t_values):
    hist = windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)
    return np.hstack([hist, np.tile(np.column_stack([hour, weekend]), (n_sq, 1))]).astype(np.float32)


internet = logs[:, :, 2]
by_day = internet[:, :WEEK].reshape(n_sq, 7, 24)
typical = np.stack([np.median(by_day[:, :5], axis=1), np.median(by_day[:, 5:], axis=1)], axis=2)


def answers(t_values):
    nxt = t_values + 1
    kind = ((nxt // 24) % 7 >= 5).astype(int)
    return {"Q1 internet": internet[:, nxt].reshape(-1), "Q2 calls": logs[:, nxt, 1].reshape(-1),
            "Q3 unusual": (np.abs(internet[:, nxt] - typical[:, nxt % 24, kind]) > np.log(2)).astype(int).reshape(-1)}


def error(question, model, X, y):
    if question == "Q3 unusual":
        p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return float(np.abs(model.predict(X) - y).mean())


X_fit, X_check = inputs(fit_t), inputs(check_t)
y_fit, y_check = answers(fit_t), answers(check_t)
n = len(X_fit)
print(f"Fit on week 1: {n:,} examples; check on week 2: {len(X_check):,}")

err = {}
for share in [1.0] + SHARES:
    rows = slice(None) if share == 1.0 else np.sort(np.random.default_rng(0).choice(n, size=int(n * share), replace=False))
    for q in y_fit:
        model = (lgb.LGBMClassifier if q == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS)
        model.fit(X_fit[rows], y_fit[q][rows])
        err[share, q] = error(q, model, X_check, y_check[q])
    print(f"  share {share:.1%} done ({time.time() - start_all:.0f} s)")

print(f"\nRaw data at each share, error relative to raw at 100% (rule: >= +5% on at least 2 of 3):\n")
print(f"{'share':>7}" + "".join(f"{q:>14}" for q in y_fit) + "   meets the rule")
chosen = None
for share in SHARES:
    worse = [err[share, q] / err[1.0, q] - 1 for q in y_fit]
    meets = sum(w >= 0.05 for w in worse) >= 2
    if meets and chosen is None:
        chosen = share
    print(f"{share:>7.1%}" + "".join(f"{w:>+14.1%}" for w in worse) + f"   {'yes' if meets else 'no'}")
print(f"\nChosen share (largest that meets the rule): {'none' if chosen is None else f'{chosen:.1%}'}")
