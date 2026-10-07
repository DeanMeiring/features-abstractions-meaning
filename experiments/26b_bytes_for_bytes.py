"""Experiment 26b: does the two-part language beat compressed raw data, byte for byte?

Part of the review after experiment 26. No new encoder, no retraining: experiment 25's
dictionary and experiment 26's setup exactly. The question: are the 16 learned meaning
characters worth more than spending the same bytes on more exact recent values?
    A  meaning + precision   experiment 26's 19-byte message (recomputed; must reproduce it)
    B  raw-18                the last 6 hours of SMS, calls and internet, 1 byte each, 18 bytes
    C  raw-9                 the last 3 hours, 9 bytes (about the two-part message bit-packed)
    D  raw data              72 numbers, the reference
B and C use exactly the precision characters' code: log(1 + x) on 256 even steps between
that activity's lowest and highest training-week value.

Bar (README, committed before this was built):
    reproduction: A gives 0.1083 (Q1) and 0.1493 (Q2), +9.4% / +7.1% vs D; otherwise stop
    "learned characters pay for their bytes": A / B - 1 whole interval below 0 on Q1 and Q2
    "compressed raw is at least as good":     A / B - 1 whole interval above 0 on Q1 or Q2
    else: too close to call, counts as not shown; no reruns
    python experiments/26b_bytes_for_bytes.py --smoke   # 400 squares, 20 trees (not a result)
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
HISTORY = 24
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
EXPERIMENT_26 = {"Q1 internet": (0.1083, 0.094), "Q2 calls": (0.1493, 0.071)}   # error, vs raw data
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
train_t = np.arange(HISTORY - 1, train_hours - 1)       # t + 1 in the training weeks (as experiment 21)
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)  # t + 1 in the test week

# The meaning characters: experiment 25's dictionary, reused as in experiment 26.
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


def recent_bytes(t_values, hours):
    """The last `hours` hours (t - hours + 1 ... t) of each activity, 1 byte each: hours * 3 bytes.
    hours = 1 is exactly experiment 26's precision characters."""
    last = np.stack([logs[:, t_values - j] for j in range(hours)], axis=2)   # (squares, t, hours, 3)
    coded = np.clip(np.round((last - low) / (high - low) * 255), 0, 255).astype(np.uint8)
    return coded.reshape(-1, hours * 3)


def extras(t_values):
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


internet = logs[:, :, 2]
by_day = internet[:, :train_hours].reshape(n_sq, 14, 24)
weekday_days = [d for d in range(14) if d % 7 < 5]
weekend_days = [d for d in range(14) if d % 7 >= 5]
typical = np.stack([np.median(by_day[:, weekday_days], axis=1), np.median(by_day[:, weekend_days], axis=1)], axis=2)


def answers(t_values):
    nxt = t_values + 1
    kind = ((nxt // 24) % 7 >= 5).astype(int)
    return {"Q1 internet": internet[:, nxt].reshape(-1), "Q2 calls": logs[:, nxt, 1].reshape(-1),
            "Q3 unusual": (np.abs(internet[:, nxt] - typical[:, nxt % 24, kind]) > np.log(2)).astype(int).reshape(-1)}


def errors(question, predicted, actual):
    if question == "Q3 unusual":
        p = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -(actual * np.log(p) + (1 - actual) * np.log(1 - p))
    return np.abs(predicted - actual)


def vectors(words):
    return sum(books[j][words[:, j]] for j in range(16)).astype(np.float32)


train_y, test_y = answers(train_t), answers(test_t)
test_blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))
m_train, m_test = meaning(train_t), meaning(test_t)
print(f"Characters ready: {len(m_train):,} training and {len(m_test):,} test windows ({time.time() - start_all:.0f} s)")


def arm_inputs(name):
    """(training inputs, test inputs) for one arm, built only when needed to keep memory down."""
    if name == "D raw data":
        return (np.hstack([history(train_t).astype(np.float32), extras(train_t)]),
                np.hstack([history(test_t).astype(np.float32), extras(test_t)]))
    if name == "A meaning + precision":
        return (np.hstack([vectors(m_train), recent_bytes(train_t, 1).astype(np.float32), extras(train_t)]),
                np.hstack([vectors(m_test), recent_bytes(test_t, 1).astype(np.float32), extras(test_t)]))
    hours = {"B raw-18": 6, "C raw-9": 3}[name]
    return (np.hstack([recent_bytes(train_t, hours).astype(np.float32), extras(train_t)]),
            np.hstack([recent_bytes(test_t, hours).astype(np.float32), extras(test_t)]))


results, predictions = {}, {}
for name in ("D raw data", "A meaning + precision", "B raw-18", "C raw-9"):
    X_train, X_test = arm_inputs(name)
    results[name] = {}
    for q in train_y:
        fitted = (lgb.LGBMClassifier if q == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS).fit(X_train, train_y[q])
        predicted = fitted.predict_proba(X_test)[:, 1] if q == "Q3 unusual" else fitted.predict(X_test)
        predictions[f"{name[0]} {q}"] = predicted.astype(np.float32)
        results[name][q] = errors(q, predicted, test_y[q])
    del X_train, X_test
    print(f"  {name} done ({time.time() - start_all:.0f} s since start)")

    if name == "A meaning + precision":   # Step 2: the reproduction check, before anything else counts.
        print("\nReproduction check (experiment 26: 0.1083 / 0.1493, +9.4% / +7.1% vs raw data):")
        same = []
        for q, (error_26, gap_26) in EXPERIMENT_26.items():
            error = results[name][q].mean()
            gap = results[name][q].mean() / results["D raw data"][q].mean() - 1
            same.append(round(error, 4) == error_26 and round(gap, 3) == gap_26)
            print(f"  {q}: {error:.4f}, {gap:+.1%} vs raw data -> {'same' if same[-1] else 'DIFFERENT'}")
        if not all(same):
            if not SMOKE:
                sys.exit("  -> DOES NOT REPRODUCE: stopping, no verdict (bar)")
            print("  -> different (expected in a smoke run; continuing just to test the code)")
        else:
            print("  -> reproduced\n")

out_file = ROOT / "data" / ("exp26b_predictions_smoke.npz" if SMOKE else "exp26b_predictions.npz")
np.savez(out_file, blocks=test_blocks, **{f"answer {q}": test_y[q] for q in test_y},
         **{k.replace(" ", "_"): v for k, v in predictions.items()})
print(f"Predictions saved to {out_file.relative_to(ROOT)}")

arms = ["A meaning + precision", "B raw-18", "C raw-9", "D raw data"]
print("\nTest-week error (Q1, Q2: mean absolute error of log(1 + x); Q3: log loss):\n")
print(f"{'':<14}" + "".join(f"{name:>24}" for name in arms))
for q in train_y:
    print(f"{q:<14}" + "".join(f"{results[name][q].mean():>24.4f}" for name in arms))
print(f"{'bytes':<14}" + "".join(f"{b:>24}" for b in ("19 (~9.5 packed)", 18, 9, 288)))


def interval(a, b, q):
    return telecom.block_interval(results[a][q], results[b][q], test_blocks)


print("\nThe bar: A / B - 1 (95% intervals over spatial blocks):")
below, above = [], []
for q in ("Q1 internet", "Q2 calls"):
    point, lo, hi = interval("A meaning + precision", "B raw-18", q)
    below.append(hi < 0)
    above.append(lo > 0)
    print(f"  {q}: {point:+.1%} [{lo:+.1%}, {hi:+.1%}]")
if all(below):
    verdict = "LEARNED CHARACTERS PAY FOR THEIR BYTES"
elif any(above):
    verdict = "COMPRESSED RAW IS AT LEAST AS GOOD"
else:
    verdict = "TOO CLOSE TO CALL: not shown; no reruns"
print(f"Verdict: {verdict}")

print("\nReported, no bar (relative error, 95% interval):")
for a, b in (("A meaning + precision", "C raw-9"), ("B raw-18", "D raw data"), ("C raw-9", "D raw data")):
    for q in train_y:
        point, lo, hi = interval(a, b, q)
        print(f"  {a[0]} vs {b[0]}, {q}: {point:+.1%} [{lo:+.1%}, {hi:+.1%}]")
for q in ("Q3 unusual",):
    point, lo, hi = interval("A meaning + precision", "B raw-18", q)
    print(f"  A vs B, {q}: {point:+.1%} [{lo:+.1%}, {hi:+.1%}]")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
