"""Experiment 23: experiment 21's forecasting test, unchanged, on a healthy dictionary (experiment 22b's).

No dictionary is trained here: it loads experiment 22b's kept dictionary and refuses to run unless
22b passed its health gate. Everything below is experiment 21's description and bar, unchanged.

Experiment 21: do correction words close the forecasting gap? (a mechanism check)

Experiment 20's 8 set words blurred the exact recent level and lost badly on next-hour
forecasting. Here the words are coarse to fine (fam/residual_words.py): word 1 predicts
the 24-hour pattern, and each further word encodes what's still missing. Messages of
k = 1, 2, 4, 8 and 16 words are compared with raw data, everything else exactly as in
experiment 20, with every arm trained on 100% of the training examples.

Bar (README, committed before running):
    pass:  at k = 16, on both Q1 (internet) and Q2 (calls), words/raw - 1 has its whole
           95% interval below +10%, AND words@16/words@1 - 1 has its whole interval below 0
    fail:  at k = 16, on Q1 or Q2, the whole interval is above +10% -> drop the forecasting fix
    else:  too close to call, counts as not shown; no reruns with tweaks
Intervals resample the 100 spatial blocks of 10 x 10 squares.
    python experiments/21_correction_words.py --smoke   # tiny run to check the code (not a result)
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
MODEL_FILE = ROOT / "data" / "models" / ("telecom_correction_words_exp22b_smoke.pt" if SMOKE else "telecom_correction_words_exp22b.pt")
HISTORY, KS = 24, [1, 2, 4, 8, 16]
LR = 5e-4  # run 2: 0.002 broke the training in run 1; experiment 21a chose 0.0005 on training data only
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

# Experiment 20's data, split, histories and answers.
counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))
spread = logs[:, :train_hours].std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


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


train_y, test_y = answers(train_t), answers(test_t)
test_blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))

# Step 1: experiment 22b's dictionary, only if it passed its health gate.
import json  # noqa: E402
gate = json.loads(MODEL_FILE.with_suffix(".gate.json").read_text())
if not gate["healthy"]:
    sys.exit("Experiment 22b's dictionary failed its health gate: experiment 23 doesn't run (stop rule).")
words_model = residual_words.CorrectionWords()
words_model.load_state_dict(torch.load(MODEL_FILE))
words_model.eval()
print(f"Loaded {MODEL_FILE.name}: passed its health gate (kept 1 / 4 / 16 words: "
      + " / ".join(f"{e:.3f}" for e in gate["kept_1_4_16"]) + ")")


def all_words(t_values):
    """(squares * len(t), 16) word numbers, translated a day of hours at a time to keep memory low."""
    out = np.zeros((n_sq, len(t_values), 16), dtype=np.uint8)
    for i in range(0, len(t_values), 24):
        chunk = t_values[i:i + 24]
        out[:, i:i + len(chunk)] = residual_words.words_of(words_model, standardise(history(chunk))).reshape(n_sq, len(chunk), 16)
    return out.reshape(-1, 16)


t = time.time()
w_train, w_test = all_words(train_t), all_words(test_t)
books = words_model.codebooks.detach().numpy()
print(f"Words for every square-hour ready ({time.time() - t:.0f} s)")


def prefix(words, k):
    return sum(books[j][words[:, j]] for j in range(k)).astype(np.float32)


# Rebuild error per k, on the test week's first day of histories (reported).
first_day = test_t[:24]
x_day = standardise(history(first_day))
w_day = residual_words.words_of(words_model, x_day)
with torch.no_grad():
    rebuild = {k: float(((words_model.decoder(torch.tensor(prefix(w_day, k))).numpy() - x_day) ** 2).mean()) for k in KS}


def errors(question, predicted, actual):
    if question == "Q3 unusual":
        p = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -(actual * np.log(p) + (1 - actual) * np.log(1 - p))
    return np.abs(predicted - actual)


def fit_and_test(X_train, X_test):
    out, secs = {}, {}
    for q in train_y:
        model = (lgb.LGBMClassifier if q == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS)
        t = time.time()
        model.fit(X_train, train_y[q])
        secs[q] = time.time() - t
        predicted = model.predict_proba(X_test)[:, 1] if q == "Q3 unusual" else model.predict(X_test)
        out[q] = errors(q, predicted, test_y[q])
    return out, secs


# Step 2: raw data, then words with k = 1, 2, 4, 8, 16.
results, seconds = {}, {}
X_train = np.hstack([history(train_t).astype(np.float32), extras(train_t)])
X_test = np.hstack([history(test_t).astype(np.float32), extras(test_t)])
results["raw"], seconds["raw"] = fit_and_test(X_train, X_test)
del X_train, X_test
print(f"  raw done ({time.time() - start_all:.0f} s since start)")
for k in KS:
    X_train = np.hstack([prefix(w_train, k), extras(train_t)])
    X_test = np.hstack([prefix(w_test, k), extras(test_t)])
    results[k], seconds[k] = fit_and_test(X_train, X_test)
    del X_train, X_test
    print(f"  words, k = {k} done ({time.time() - start_all:.0f} s since start)")

# Step 3: the curve, and the verdict against the bar fixed before running.
print("\nTest-week error by message length (Q1, Q2: mean absolute error of log(1 + x); Q3: log loss):\n")
print(f"{'':<14}{'raw':>8}" + "".join(f"{'k=' + str(k):>9}" for k in KS))
for q in train_y:
    print(f"{q:<14}{results['raw'][q].mean():>8.4f}" + "".join(f"{results[k][q].mean():>9.4f}" for k in KS))
print(f"{'rebuild error':<14}{'':>8}" + "".join(f"{rebuild[k]:>9.3f}" for k in KS) + "   (standardised, test week's first day)")
print(f"{'bytes':<14}{'288':>8}" + "".join(f"{k:>9}" for k in KS))

print("\nWords vs raw data (relative error, 95% interval over spatial blocks):")
verdicts = []
for q in ("Q1 internet", "Q2 calls", "Q3 unusual"):
    line = []
    for k in KS:
        v, lo, hi = telecom.block_interval(results[k][q], results["raw"][q], test_blocks)
        line.append(f"k={k} {v:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    print(f"  {q}: " + " | ".join(line))
    if q != "Q3 unusual":
        gap = telecom.block_interval(results[16][q], results["raw"][q], test_blocks)
        better = telecom.block_interval(results[16][q], results[1][q], test_blocks)
        verdicts.append((q, gap, better))

print("\nVerdict (k = 16, Q1 and Q2):")
passes = all(gap[2] < 0.10 and better[2] < 0 for _, gap, better in verdicts)
fails = any(gap[1] > 0.10 for _, gap, _ in verdicts)
for q, gap, better in verdicts:
    print(f"  {q}: words@16 vs raw {gap[0]:+.1%} [{gap[1]:+.1%}, {gap[2]:+.1%}] (bar: whole interval below +10%); "
          f"words@16 vs words@1 {better[0]:+.1%} [{better[1]:+.1%}, {better[2]:+.1%}] (bar: below 0)")
print("  -> " + ("PASS: corrections close the gap" if passes else
                 "FAIL: corrections don't close the gap; drop the forecasting fix" if fails else
                 "TOO CLOSE TO CALL: not shown; no reruns"))
print("\nTraining seconds: " + ", ".join(f"{a}: {sum(seconds[a].values()):.0f}" for a in ["raw"] + KS))
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
