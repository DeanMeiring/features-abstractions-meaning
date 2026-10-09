"""Experiment 28: words on top of raw data, not instead of it.

The developer's vision: raw data stays, and the language adds shared context, so features
don't have to be engineered per model. Never tested on telecom. Four arms, all with the hour
of day and weekend flag of the target hour:
    R      raw data, the past 24 hours (72 numbers)
    R+W    raw data + the 16 meaning characters (experiment 25's dictionary, as their vectors)
    R+H    raw data + hand-built context: per activity, the square's typical value at the target
           hour and how far hour t is from its typical value (6 numbers; typical = the median
           over the training weeks' days of the same kind, weekday or weekend)
    R+H+W  both (reported)
Four questions: next-hour (t + 1) and next-day (t + 24) internet and calls, mean absolute error.

Bar (README, committed before this was built):
    pass:  (1) R+W / R - 1 whole interval below 0 on >= 3 of 4, and
           (2) R+W / R+H - 1 whole interval below +1% on >= 3 of 4
    (1) without (2): "the words add context, but a hand-built feature does as well or better"
    fail:  R+W / R - 1 whole interval at or above 0 on >= 2 of 4
    else:  too close to call, counts as not shown; no reruns
    python experiments/28_raw_plus_words.py --smoke   # 400 squares, 20 trees (not a result)
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
ARMS = ["R", "R+W", "R+H", "R+H+W"]
start_all = time.time()

counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
full_logs = np.log1p(counts[:, :train_hours])           # the dictionary's statistics: all squares, training weeks
mean = full_logs.mean(axis=(0, 1))
spread = full_logs.std(axis=(0, 1))
del full_logs
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)

model = residual_words.CorrectionWords(future=18)
model.load_state_dict(torch.load(ROOT / "data" / "models" / "telecom_predictive_words.pt"))
model.eval()
books = model.codebooks.detach().numpy()

# Hand-built context: each square's typical value per hour of day, weekday or weekend, training weeks only.
by_day = logs[:, :train_hours].reshape(n_sq, 14, 24, 3)
weekday_days = [d for d in range(14) if d % 7 < 5]
weekend_days = [d for d in range(14) if d % 7 >= 5]
typical = np.stack([np.median(by_day[:, weekday_days], axis=1), np.median(by_day[:, weekend_days], axis=1)],
                   axis=2)                               # (squares, 24 hours, weekday/weekend, 3 activities)


def kind(hours):
    return ((hours // 24) % 7 >= 5).astype(int)


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3).astype(np.float32)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


@torch.no_grad()
def word_vectors(t_values):
    """The 16 meaning characters per window (past 24 hours only), as the sum of their vectors."""
    out = []
    for i in range(0, len(t_values), 24):
        z = model.encoder(torch.tensor(standardise(history(t_values[i:i + 24]))))
        words = model.quantize(z)[0].numpy()
        out.append(sum(books[j][words[:, j]] for j in range(16)).reshape(n_sq, -1, books.shape[2]))
    return np.concatenate(out, axis=1).reshape(-1, books.shape[2]).astype(np.float32)


def hand_built(t_values, ahead):
    """Per activity: typical value at the target hour, and hour t minus its typical value (6 numbers)."""
    target = t_values + ahead
    at_target = typical[:, target % 24, kind(target)]                    # (squares, t, 3)
    deviation = logs[:, t_values] - typical[:, t_values % 24, kind(t_values)]
    return np.concatenate([at_target, deviation], axis=2).reshape(-1, 6).astype(np.float32)


def extras(t_values, ahead):
    target = t_values + ahead
    return np.tile(np.column_stack([(target % 24).astype(np.float32), kind(target).astype(np.float32)]), (n_sq, 1))


results, predictions = {}, {}
for horizon, ahead in (("next hour", 1), ("next day", 24)):
    train_t = np.arange(HISTORY - 1, train_hours - ahead)                # targets in the training weeks
    test_t = np.arange(train_hours - ahead, telecom.HOURS - ahead)       # targets in the test week
    parts = {}
    for split, t in (("train", train_t), ("test", test_t)):
        parts[split] = {"R": history(t), "W": word_vectors(t), "H": hand_built(t, ahead), "x": extras(t, ahead)}
    answers = {split: {"internet": logs[:, t + ahead, 2].reshape(-1), "calls": logs[:, t + ahead, 1].reshape(-1)}
               for split, t in (("train", train_t), ("test", test_t))}
    blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))
    print(f"{horizon}: inputs ready, {len(parts['train']['R']):,} training and {len(parts['test']['R']):,} test windows "
          f"({time.time() - start_all:.0f} s)")
    for arm in ARMS:
        pieces = arm.split("+") + ["x"]                                  # e.g. "R+H" -> R, H, extras
        X_train = np.hstack([parts["train"][p] for p in pieces])
        X_test = np.hstack([parts["test"][p] for p in pieces])
        for activity in ("internet", "calls"):
            q = f"{horizon} {activity}"
            predicted = lgb.LGBMRegressor(**SETTINGS).fit(X_train, answers["train"][activity]).predict(X_test)
            predictions[f"{arm}_{q}".replace(" ", "_")] = predicted.astype(np.float32)
            results.setdefault(q, {"blocks": blocks})[arm] = np.abs(predicted - answers["test"][activity])
        del X_train, X_test
        print(f"  {arm} done ({time.time() - start_all:.0f} s since start)")
    del parts

out_file = ROOT / "data" / ("exp28_predictions_smoke.npz" if SMOKE else "exp28_predictions.npz")
np.savez(out_file, **predictions)
print(f"Predictions saved to {out_file.relative_to(ROOT)}")

print("\nTest-week error (mean absolute error of log(1 + x)):\n")
print(f"{'':<18}" + "".join(f"{arm:>10}" for arm in ARMS))
for q, res in results.items():
    print(f"{q:<18}" + "".join(f"{res[arm].mean():>10.4f}" for arm in ARMS))


def interval(q, a, b):
    return telecom.block_interval(results[q][a], results[q][b], results[q]["blocks"])


print("\nThe bar (95% intervals over spatial blocks):")
gains, keeps, no_gain = 0, 0, 0
for q in results:
    w_vs_r, w_vs_h = interval(q, "R+W", "R"), interval(q, "R+W", "R+H")
    gains += w_vs_r[2] < 0
    keeps += w_vs_h[2] < 0.01
    no_gain += w_vs_r[1] >= 0
    print(f"  {q:<18} R+W vs R {w_vs_r[0]:+.1%} [{w_vs_r[1]:+.1%}, {w_vs_r[2]:+.1%}] (needs < 0)   "
          f"R+W vs R+H {w_vs_h[0]:+.1%} [{w_vs_h[1]:+.1%}, {w_vs_h[2]:+.1%}] (needs < +1%)")
print(f"  (1) words beat raw alone on {gains} of 4 (needs 3); (2) words within 1% of hand-built on {keeps} of 4 (needs 3)")
if gains >= 3 and keeps >= 3:
    verdict = "PASS: the language replaces per-model feature engineering"
elif no_gain >= 2:
    verdict = "FAIL: the words add no context on top of raw data"
elif gains >= 3:
    verdict = "(1) WITHOUT (2): the words add context, but a hand-built feature does as well or better"
else:
    verdict = "TOO CLOSE TO CALL: not shown; no reruns"
print(f"Verdict: {verdict}")

print("\nReported, no bar:")
for q in results:
    h_vs_r, all_vs_h = interval(q, "R+H", "R"), interval(q, "R+H+W", "R+H")
    print(f"  {q:<18} R+H vs R {h_vs_r[0]:+.1%} [{h_vs_r[1]:+.1%}, {h_vs_r[2]:+.1%}]   "
          f"R+H+W vs R+H {all_vs_h[0]:+.1%} [{all_vs_h[1]:+.1%}, {all_vs_h[2]:+.1%}]")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
