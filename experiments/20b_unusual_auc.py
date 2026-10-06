"""Experiment 20b: is experiment 20's unusual-activity signal meaning or calibration?

In experiment 20, on Q3 (unusual internet activity next hour), words at 100% had a
lower log loss than raw data at 100% (-21%). Log loss rewards both ranking the right
hours as likely and guessing how often unusual hours happen, and the test week had
3x more unusual hours than training. AUC measures ranking only. So:
    words still better on AUC  -> the signal is about meaning
    gap gone                   -> it was calibration under the shifted test week

Experiment 20 saved no predictions, so its two Q3 models are refit with exactly its
inputs, settings, saved dictionary and seeds; the refit must reproduce its log
losses (raw 0.1912, words 0.1515) to 3 decimals before any verdict.

Bar fixed in the README before computing. Needs experiment 20 first (its dictionary).
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
from sklearn.metrics import roc_auc_score

from fam import telecom, window_words

EXPECTED = {"raw": 0.1912, "words": 0.1515}       # experiment 20's Q3 log losses at 100%
HISTORY, SHARE = 24, 0.02
SETTINGS = dict(n_estimators=300, learning_rate=0.1, num_leaves=63, random_state=0, n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

# Exactly experiment 20's data, inputs and answers (copied, not changed).
logs = np.log1p(telecom.load_hourly())
n_sq = len(logs)
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))
spread = logs[:, :train_hours].std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)

words_model = window_words.WindowWords()
words_model.load_state_dict(torch.load(ROOT / "data" / "models" / "telecom_words.pt"))
words_model.eval()
table = words_model.codebook.detach().numpy()


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


def extras(t_values):
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


def raw_inputs(t_values):
    return np.hstack([history(t_values).astype(np.float32), extras(t_values)])


def word_inputs(t_values):
    words = np.zeros((n_sq, len(t_values), 8), dtype=np.uint8)
    for i in range(0, len(t_values), 24):
        chunk = t_values[i:i + 24]
        words[:, i:i + len(chunk)] = window_words.words_of(words_model, standardise(history(chunk))).reshape(n_sq, len(chunk), 8)
    words = np.sort(words.reshape(-1, 8), axis=1)
    return np.hstack([table[words].reshape(len(words), -1), extras(t_values)])


internet = logs[:, :, 2]
by_day = internet[:, :train_hours].reshape(n_sq, 14, 24)
weekday_days = [d for d in range(14) if d % 7 < 5]
weekend_days = [d for d in range(14) if d % 7 >= 5]
typical = np.stack([np.median(by_day[:, weekday_days], axis=1), np.median(by_day[:, weekend_days], axis=1)], axis=2)


def unusual(t_values):
    nxt = t_values + 1
    kind = ((nxt // 24) % 7 >= 5).astype(int)
    return (np.abs(internet[:, nxt] - typical[:, nxt % 24, kind]) > np.log(2)).astype(int).reshape(-1)


y_train, y_test = unusual(train_t), unusual(test_t)
few = np.sort(np.random.default_rng(0).choice(len(train_t) * n_sq, size=int(len(train_t) * n_sq * SHARE), replace=False))
test_blocks = np.repeat(telecom.spatial_blocks(), len(test_t))

# Refit the Q3 models (and the 2% ones, reported only).
prob = {}
for name, build in (("raw", raw_inputs), ("words", word_inputs)):
    X_train, X_test = build(train_t), build(test_t)
    for share, rows in (("100%", slice(None)), ("2%", few)):
        model = lgb.LGBMClassifier(**SETTINGS).fit(X_train[rows], y_train[rows])
        prob[name, share] = model.predict_proba(X_test)[:, 1]
    del X_train, X_test
    print(f"{name} refit ({time.time() - start_all:.0f} s since start)")


def log_loss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# Reproduction first.
reproduced = {name: log_loss(prob[name, "100%"], y_test) for name in EXPECTED}
same = all(round(reproduced[n], 3) == round(EXPECTED[n], 3) for n in EXPECTED)
print("\nReproduction: " + ", ".join(f"{n} {reproduced[n]:.4f} (experiment 20: {EXPECTED[n]:.4f})" for n in EXPECTED)
      + f"  -> {'reproduced' if same else 'NOT REPRODUCED: no verdict'}")
if not same:
    sys.exit()


def block_auc_diff(p_words, p_raw, y, blocks, draws=2000, seed=0):
    """AUC(words) - AUC(raw), with a 95% interval from resampling whole spatial blocks.

    Each draw weights every example by how often its block was drawn; the weighted AUC is
    computed from one sort per model (no ties in practice: the scores are continuous).
    """
    n_blocks = blocks.max() + 1
    sorted_by = {"w": np.argsort(p_words, kind="stable"), "r": np.argsort(p_raw, kind="stable")}
    rng = np.random.default_rng(seed)

    def weighted_auc(order, weight):
        pos, w = y[order], weight[order]
        neg_w = w * (1 - pos)
        below = np.cumsum(neg_w) - neg_w             # negative weight ranked strictly below each example
        return float((w * pos * below).sum() / ((w * pos).sum() * neg_w.sum()))

    diffs = []
    for _ in range(draws):
        counts = np.bincount(rng.integers(n_blocks, size=n_blocks), minlength=n_blocks).astype(float)
        weight = counts[blocks]
        diffs.append(weighted_auc(sorted_by["w"], weight) - weighted_auc(sorted_by["r"], weight))
    ones = np.ones(len(y))
    point = weighted_auc(sorted_by["w"], ones) - weighted_auc(sorted_by["r"], ones)
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


auc = {key: roc_auc_score(y_test, p) for key, p in prob.items()}
diff, low, high = block_auc_diff(prob["words", "100%"], prob["raw", "100%"], y_test, test_blocks)

print(f"\nQ3 unusual activity, test week ({y_test.mean():.1%} of hours are unusual):\n")
print(f"{'arm':<14}{'AUC':>8}{'mean predicted':>17}")
for name in ("raw", "words"):
    for share in ("100%", "2%"):
        print(f"{name + ' ' + share:<14}{auc[name, share]:>8.4f}{prob[name, share].mean():>17.1%}")

print(f"\nAUC, words minus raw (both at 100%): {diff:+.4f} [{low:+.4f}, {high:+.4f}] (95%, spatial blocks)")
print("Verdict: " + ("MEANING: the words rank unusual hours better; the signal is about meaning, not only calibration"
                     if low > 0 else
                     "CALIBRATION: no evidence of meaning; experiment 20's log-loss gap came from calibration under the shifted test week"))
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
