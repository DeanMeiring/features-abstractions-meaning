"""Experiment 27: next-day forecasting, the learned language against a strong equal-bytes raw code.

The final telecom test (decided 2026-10-07 after the review). Experiment 26b showed the learned
characters add nothing on next-hour forecasting beyond more exact recent values. Here the
question is the same hour tomorrow (t + 24), where a day's type should matter more, and the
raw control is the best simple code for that question, not the last 6 hours again.
    A  meaning + precision   experiment 26's 19-byte message (16 meaning characters + hour t)
    B  strong raw-18         hours t, t-1, t-2 and t-23, t-24, t-25, 1 byte each (18 bytes)
    C  same hour yesterday   hour t only (3 bytes)
    D  raw data              the past 24 hours, 72 numbers
B and C use the precision characters' code: log(1 + x) on 256 even steps between that
activity's lowest and highest training-week value. Experiment 25's dictionary, unchanged.

Bar (README, committed before this was built):
    pass:      A / B - 1 whole 95% interval below 0 on both internet and calls
    otherwise: the telecom chapter closes; no reruns, no tweaks
    python experiments/27_next_day.py --smoke   # 400 squares, 20 trees (not a result)
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

SMOKE = "--smoke" in sys.argv
HISTORY, AHEAD = 24, 24
B_HOURS = [0, 1, 2, 23, 24, 25]                          # hours before t that arm B codes
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
full_logs = np.log1p(counts[:, :train_hours])           # the dictionary's statistics: all squares, training weeks
mean = full_logs.mean(axis=(0, 1))
spread = full_logs.std(axis=(0, 1))
low, high = full_logs.min(axis=(0, 1)), full_logs.max(axis=(0, 1))   # the byte code's range, training weeks only
del full_logs
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
first_t = max(B_HOURS)                                   # B needs hour t - 25
train_t = np.arange(first_t, train_hours - AHEAD)        # targets t + 24 in the training weeks
test_t = np.arange(train_hours - AHEAD, telecom.HOURS - AHEAD)   # targets t + 24 in the test week

model = residual_words.CorrectionWords(future=18)
model.load_state_dict(torch.load(ROOT / "data" / "models" / "telecom_predictive_words.pt"))
model.eval()
books = model.codebooks.detach().numpy()


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


@torch.no_grad()
def meaning(t_values):
    """16 meaning characters per window (made from the past 24 hours only)."""
    out = []
    for i in range(0, len(t_values), 24):
        z = model.encoder(torch.tensor(standardise(history(t_values[i:i + 24]))))
        out.append(model.quantize(z)[0].numpy().astype(np.uint8).reshape(n_sq, -1, 16))
    return np.concatenate(out, axis=1).reshape(-1, 16)


def coded(t_values, hours_back):
    """Each activity at hours t - h (h in hours_back), in the precision characters' 1-byte code."""
    values = np.stack([logs[:, t_values - h] for h in hours_back], axis=2)   # (squares, t, hours, 3)
    code = np.clip(np.round((values - low) / (high - low) * 255), 0, 255).astype(np.uint8)
    return code.reshape(-1, len(hours_back) * 3)


def extras(t_values):
    """Hour of day and weekend flag of the target hour (t + 24)."""
    target = t_values + AHEAD
    hour = (target % 24).astype(np.float32)
    weekend = ((target // 24) % 7 >= 5).astype(np.float32)
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


def answers(t_values):
    target = t_values + AHEAD
    return {"Q1 internet": logs[:, target, 2].reshape(-1), "Q2 calls": logs[:, target, 1].reshape(-1)}


def vectors(words):
    return sum(books[j][words[:, j]] for j in range(16)).astype(np.float32)


train_y, test_y = answers(train_t), answers(test_t)
test_blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))
m_train, m_test = meaning(train_t), meaning(test_t)
print(f"Characters ready: {len(m_train):,} training and {len(m_test):,} test windows ({time.time() - start_all:.0f} s)")


def arm_inputs(name):
    """(training inputs, test inputs) for one arm, built only when needed to keep memory down."""
    if name == "A meaning + precision":
        return (np.hstack([vectors(m_train), coded(train_t, [0]).astype(np.float32), extras(train_t)]),
                np.hstack([vectors(m_test), coded(test_t, [0]).astype(np.float32), extras(test_t)]))
    make = {"B strong raw-18": lambda t: coded(t, B_HOURS),
            "C same hour yesterday": lambda t: coded(t, [0]),
            "D raw data": history}[name]
    return (np.hstack([make(train_t).astype(np.float32), extras(train_t)]),
            np.hstack([make(test_t).astype(np.float32), extras(test_t)]))


arms = ["A meaning + precision", "B strong raw-18", "C same hour yesterday", "D raw data"]
results, predictions = {}, {}
for name in arms:
    X_train, X_test = arm_inputs(name)
    results[name] = {}
    for q in train_y:
        predicted = lgb.LGBMRegressor(**SETTINGS).fit(X_train, train_y[q]).predict(X_test)
        predictions[f"{name[0]}_{q.replace(' ', '_')}"] = predicted.astype(np.float32)
        results[name][q] = np.abs(predicted - test_y[q])
    del X_train, X_test
    print(f"  {name} done ({time.time() - start_all:.0f} s since start)")

naive = {"Q1 internet": np.abs(logs[:, test_t, 2].reshape(-1) - test_y["Q1 internet"]),
         "Q2 calls": np.abs(logs[:, test_t, 1].reshape(-1) - test_y["Q2 calls"])}

out_file = ROOT / "data" / ("exp27_predictions_smoke.npz" if SMOKE else "exp27_predictions.npz")
np.savez(out_file, blocks=test_blocks, **{f"answer_{q.replace(' ', '_')}": test_y[q] for q in test_y}, **predictions)
print(f"Predictions saved to {out_file.relative_to(ROOT)}")

print("\nTest-week error, same hour tomorrow (mean absolute error of log(1 + x)):\n")
print(f"{'':<14}" + "".join(f"{name:>24}" for name in arms) + f"{'naive (no model)':>20}")
for q in train_y:
    print(f"{q:<14}" + "".join(f"{results[name][q].mean():>24.4f}" for name in arms) + f"{naive[q].mean():>20.4f}")
print(f"{'bytes':<14}" + "".join(f"{b:>24}" for b in ("19 (~9.5 packed)", 18, 3, 288)) + f"{3:>20}")

print("\nThe bar: A / B - 1 (95% intervals over spatial blocks):")
wins = []
for q in train_y:
    point, lo, hi = telecom.block_interval(results[arms[0]][q], results[arms[1]][q], test_blocks)
    wins.append(hi < 0)
    print(f"  {q}: {point:+.1%} [{lo:+.1%}, {hi:+.1%}]")
print("Verdict: " + ("PASS: the learned part carries the day's type" if all(wins) else
                     "NOT A PASS: the telecom chapter closes (bar)"))

print("\nReported, no bar (relative error, 95% interval):")
for a, b in ((0, 2), (0, 3), (1, 3), (2, 3)):
    for q in train_y:
        point, lo, hi = telecom.block_interval(results[arms[a]][q], results[arms[b]][q], test_blocks)
        print(f"  {arms[a][0]} vs {arms[b][0]}, {q}: {point:+.1%} [{lo:+.1%}, {hi:+.1%}]")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
