"""Experiment 26: does a two-part language (meaning + precision characters) close the forecasting gap?

The developer's idea after experiment 25, like a Chinese character with a meaning part and a
part that pins down exactly which one:
    meaning characters    16 per 24-hour window: experiment 25's predictive dictionary, reused as it is
    precision characters  3 per window: the last hour's log(1 + x) for SMS, calls and internet,
                          each mapped onto 256 even steps between that activity's lowest and
                          highest training-week value (a fixed code, 1 byte each, not learned)
A message is 16 + 3 = 19 bytes, against 288 bytes of raw history. Data, questions, model and
split as experiment 21. Arms: raw data, meaning only, precision only, meaning + precision.

Bar (README, committed before this was built):
    pass:  (1) meaning + precision within +10% of raw data on both Q1 (internet) and Q2 (calls),
           whole 95% interval over spatial blocks, AND (2) meaning + precision at least 5% better
           than precision only on both (whole interval of the ratio - 1 below -5%)
    fail:  (1) clearly missed on Q1 or Q2 (whole interval above +10%)
    (1) without (2): "the precision characters do the work; the meaning characters add too little"
    else:  too close to call, counts as not shown; no reruns with tweaks
    python experiments/26_two_part_language.py --smoke   # 400 squares, 20 trees (not a result)
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
start_all = time.time()

counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
full_logs = np.log1p(counts[:, :train_hours])           # the dictionary's statistics: all squares, training weeks
mean = full_logs.mean(axis=(0, 1))
spread = full_logs.std(axis=(0, 1))
low, high = full_logs.min(axis=(0, 1)), full_logs.max(axis=(0, 1))   # the precision code's range, training weeks only
del full_logs
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)       # t + 1 in the training weeks (as experiment 21)
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)  # t + 1 in the test week

# The meaning characters: experiment 25's dictionary, reused (it passed its health gate there).
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


def precision(t_values):
    """3 precision characters per window: the last hour (t itself) of each activity, 1 byte each."""
    last = logs[:, t_values].reshape(-1, 3)
    return np.clip(np.round((last - low) / (high - low) * 255), 0, 255).astype(np.uint8)


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


train_y, test_y = answers(train_t), answers(test_t)
test_blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))
m_train, m_test = meaning(train_t), meaning(test_t)
p_train, p_test = precision(train_t), precision(test_t)
print(f"Characters ready: {len(m_train):,} training and {len(m_test):,} test windows ({time.time() - start_all:.0f} s)")


def vectors(words):
    return sum(books[j][words[:, j]] for j in range(16)).astype(np.float32)


inputs = {
    "raw data": (np.hstack([history(train_t).astype(np.float32), extras(train_t)]),
                 np.hstack([history(test_t).astype(np.float32), extras(test_t)])),
    "meaning only": (np.hstack([vectors(m_train), extras(train_t)]), np.hstack([vectors(m_test), extras(test_t)])),
    "precision only": (np.hstack([p_train.astype(np.float32), extras(train_t)]), np.hstack([p_test.astype(np.float32), extras(test_t)])),
    "meaning + precision": (np.hstack([vectors(m_train), p_train.astype(np.float32), extras(train_t)]),
                            np.hstack([vectors(m_test), p_test.astype(np.float32), extras(test_t)])),
}
results = {}
for name, (X_train, X_test) in inputs.items():
    results[name] = {}
    for q in train_y:
        fitted = (lgb.LGBMClassifier if q == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS).fit(X_train, train_y[q])
        results[name][q] = errors(q, fitted.predict_proba(X_test)[:, 1] if q == "Q3 unusual" else fitted.predict(X_test), test_y[q])
    print(f"  {name} done ({time.time() - start_all:.0f} s since start)")

print("\nTest-week error (Q1, Q2: mean absolute error of log(1 + x); Q3: log loss):\n")
print(f"{'':<14}" + "".join(f"{name:>22}" for name in inputs))
for q in train_y:
    print(f"{q:<14}" + "".join(f"{results[name][q].mean():>22.4f}" for name in inputs))
print(f"{'bytes':<14}" + "".join(f"{b:>22}" for b in (288, 16, 3, 19)))

print("\nThe bar (95% intervals over spatial blocks):")
part1, part2, clear_fail = [], [], []
for q in ("Q1 internet", "Q2 calls"):
    vs_raw = telecom.block_interval(results["meaning + precision"][q], results["raw data"][q], test_blocks)
    vs_precision = telecom.block_interval(results["meaning + precision"][q], results["precision only"][q], test_blocks)
    precision_vs_raw = telecom.block_interval(results["precision only"][q], results["raw data"][q], test_blocks)
    part1.append(vs_raw[2] < 0.10)
    part2.append(vs_precision[2] < -0.05)
    clear_fail.append(vs_raw[1] > 0.10)
    print(f"  {q}: meaning+precision vs raw {vs_raw[0]:+.1%} [{vs_raw[1]:+.1%}, {vs_raw[2]:+.1%}] (needs whole interval < +10%)")
    print(f"  {'':<{len(q)}}  meaning+precision vs precision only {vs_precision[0]:+.1%} [{vs_precision[1]:+.1%}, "
          f"{vs_precision[2]:+.1%}] (needs whole interval < -5%)")
    print(f"  {'':<{len(q)}}  (reported) precision only vs raw {precision_vs_raw[0]:+.1%} [{precision_vs_raw[1]:+.1%}, {precision_vs_raw[2]:+.1%}]")
if all(part1) and all(part2):
    verdict = "PASS: the two-part language closes the gap, and the meaning characters earn their place"
elif any(clear_fail):
    verdict = "FAIL: the gap stays"
elif all(part1):
    verdict = "PART (1) ONLY: the precision characters do the work; the meaning characters add too little"
else:
    verdict = "TOO CLOSE TO CALL: not shown; no reruns"
print(f"Verdict: {verdict}")

# Reported: bits actually used per character, and the message size if each took only what it needs.
def bits_of(column):
    p = np.bincount(column, minlength=256) / len(column)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


meaning_bits = [bits_of(m_test[:, k]) for k in range(16)]
precision_bits = [bits_of(p_test[:, k]) for k in range(3)]
total = sum(meaning_bits) + sum(precision_bits)
print(f"\nBits used: meaning characters {sum(meaning_bits):.1f} of 128, precision characters "
      + " / ".join(f"{b:.1f}" for b in precision_bits) + f" of 8 each; whole message {total:.0f} bits "
      f"= about {total / 8:.1f} bytes if each character took only the bits it needs (19 bytes as stored)")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
