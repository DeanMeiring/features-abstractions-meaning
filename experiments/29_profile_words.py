"""Experiment 29: can learned profile words replace a hand-built feature?

Experiment 28 showed words made from the model's own 24-hour window add nothing on top of raw
data, while a hand-built feature from two weeks of each square's history helps 4-14%. Here the
words carry that longer history: a profile dictionary reads each square's raw two-week history
(all 336 training-week hours x 3 activities, 1,008 numbers) and writes 16 profile words per
square, once. (Dated change before the official run: first designed on the square's typical
weekday and weekend day, the same medians the hand-built feature uses; see the README.)
    1. train the profile dictionary on 9,000 squares, check on the other 1,000, stopping when the
       check error hasn't improved for 20 passes (at most 300; best pass kept), then the health gate
    2. forecast with four arms, all with hour of day and weekend flag of the target hour:
       R raw data (72) | R+P + the square's 16 profile words | R+H + the hand-built feature (6) | R+H+P
       on next-hour and next-day internet and calls (mean absolute error of log(1 + x))
Bar (README, committed before this was built):
    pass:  (1) R+P / R - 1 whole interval below 0 on >= 3 of 4, and
           (2) R+P / R+H - 1 whole interval below +1% on >= 3 of 4
    (1) without (2): "the profile words add the square's context, but less than the hand-built feature"
    fail:  R+P / R+H - 1 whole interval above +1% on >= 2 of 4, and (1) not met
    else:  too close to call, counts as not shown; no reruns
    python experiments/29_profile_words.py --smoke   # full dictionary training, forecasting on 400 squares, 20 trees
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
MODEL_FILE = ROOT / "data" / "models" / ("telecom_profile_words_smoke.pt" if SMOKE else "telecom_profile_words.pt")
HISTORY, EPOCHS, BATCH, PATIENCE = 24, 300, 256, 20
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
ARMS = ["R", "R+P", "R+H", "R+H+P"]
start_all = time.time()

all_logs = np.log1p(telecom.load_hourly())                # (10,000 squares, hours, 3)
train_hours = telecom.TRAIN_HOURS
mean = all_logs[:, :train_hours].mean(axis=(0, 1))
spread = all_logs[:, :train_hours].std(axis=(0, 1))


def kind(hours):
    return ((hours // 24) % 7 >= 5).astype(int)


# Each square's typical weekday and weekend day, training weeks only: (10,000, 24 hours, 2 kinds, 3 activities).
by_day = all_logs[:, :train_hours].reshape(len(all_logs), 14, 24, 3)
weekday_days = [d for d in range(14) if d % 7 < 5]
weekend_days = [d for d in range(14) if d % 7 >= 5]
typical_all = np.stack([np.median(by_day[:, weekday_days], axis=1), np.median(by_day[:, weekend_days], axis=1)], axis=2)
del by_day

# Step 1: the profile dictionary (the raw two-week history, 1,008 standardised numbers per square in, 16 words out).
profiles = ((all_logs[:, :train_hours] - mean) / spread).reshape(len(all_logs), -1).astype(np.float32)
order = np.random.default_rng(0).permutation(len(profiles))
fit_rows, check_rows = order[:9000], order[9000:]
print(f"Training the profile dictionary on {len(fit_rows):,} squares (at most {EPOCHS} passes, patience {PATIENCE}), "
      f"checking on {len(check_rows):,}...")
t0 = time.time()
model = residual_words.train(profiles[fit_rows], epochs=EPOCHS, batch_size=BATCH, lr=5e-4, check=profiles[check_rows],
                             lr_to_zero=True, keep_best=True, revive=False, patience=PATIENCE)
print(f"  took {time.time() - t0:.0f} s")
torch.save(model.state_dict(), MODEL_FILE)

history_16 = model.history[:, 15]
best_pass = int(history_16.argmin())
errors = residual_words.rebuild_errors(model, profiles[check_rows])
gate = {
    "(1) last pass within 10% of the best": history_16[-1] <= 1.10 * history_16[best_pass],
    "(2) 16-word rebuild error <= 0.5": errors[15] <= 0.5,
    "(3) error falls 1 -> 4 -> 16 words": errors[0] > errors[3] > errors[15],
}
print(f"Kept pass {best_pass + 1}; check squares, 1 / 4 / 16 words: {errors[0]:.3f} / {errors[3]:.3f} / {errors[15]:.3f}")
print("Health gate:")
for check, ok in gate.items():
    print(f"  {check}: {'yes' if ok else 'NO'}")
if not all(gate.values()):
    sys.exit("  -> FAIL: the run is invalid; no forecasting (stop rule)")
print("  -> HEALTHY")

words = residual_words.words_of(model, profiles)                         # (10,000, 16), one message per square
books = model.codebooks.detach().numpy()
profile_vectors = sum(books[j][words[:, j]] for j in range(16)).astype(np.float32)   # (10,000, 32)
bits = []
for k in range(16):
    p = np.bincount(words[:, k], minlength=256) / len(words)
    p = p[p > 0]
    bits.append(float(-(p * np.log2(p)).sum()))
print("Bits per word position: " + " ".join(f"{b:.1f}" for b in bits) + f"  (together {sum(bits):.1f} of 128)")

# Step 2: forecasting.
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
n_sq = len(squares)
logs = all_logs[squares]
typical = typical_all[squares]
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
del all_logs


def inputs(arm, t_values, ahead):
    """One arm's inputs, built in a single array to keep memory down. Rows: square by square, then hour."""
    target = t_values + ahead
    width = HISTORY * 3 + 6 * ("H" in arm) + profile_vectors.shape[1] * ("P" in arm) + 2
    X = np.empty((n_sq * len(t_values), width), dtype=np.float32)        # filled in place: no 64-bit copies
    X[:, :HISTORY * 3] = windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)
    col = HISTORY * 3
    if "H" in arm:
        X[:, col:col + 3] = typical[:, target % 24, kind(target)].reshape(-1, 3)              # typical value at the target hour
        X[:, col + 3:col + 6] = (logs[:, t_values] - typical[:, t_values % 24, kind(t_values)]).reshape(-1, 3)  # hour t vs typical
        col += 6
    if "P" in arm:
        X[:, col:col + profile_vectors.shape[1]] = np.repeat(profile_vectors[squares], len(t_values), axis=0)
        col += profile_vectors.shape[1]
    X[:, col] = np.tile(target % 24, n_sq)
    X[:, col + 1] = np.tile(kind(target), n_sq)
    return X


results, predictions = {}, {}
for horizon, ahead in (("next hour", 1), ("next day", 24)):
    train_t = np.arange(HISTORY - 1, train_hours - ahead)
    test_t = np.arange(train_hours - ahead, telecom.HOURS - ahead)
    answers = {split: {"internet": logs[:, t + ahead, 2].reshape(-1), "calls": logs[:, t + ahead, 1].reshape(-1)}
               for split, t in (("train", train_t), ("test", test_t))}
    blocks = np.repeat(telecom.spatial_blocks()[squares], len(test_t))
    for arm in ARMS:
        X_train, X_test = inputs(arm, train_t, ahead), inputs(arm, test_t, ahead)
        for activity in ("internet", "calls"):
            q = f"{horizon} {activity}"
            predicted = lgb.LGBMRegressor(**SETTINGS).fit(X_train, answers["train"][activity]).predict(X_test)
            predictions[f"{arm}_{q}".replace(" ", "_")] = predicted.astype(np.float32)
            results.setdefault(q, {"blocks": blocks})[arm] = np.abs(predicted - answers["test"][activity])
        del X_train, X_test
        print(f"  {horizon}, {arm} done ({time.time() - start_all:.0f} s since start)")

out_file = ROOT / "data" / ("exp29_predictions_smoke.npz" if SMOKE else "exp29_predictions.npz")
np.savez(out_file, **predictions)
print(f"Predictions saved to {out_file.relative_to(ROOT)}")

print("\nTest-week error (mean absolute error of log(1 + x)):\n")
print(f"{'':<18}" + "".join(f"{arm:>10}" for arm in ARMS))
for q, res in results.items():
    print(f"{q:<18}" + "".join(f"{res[arm].mean():>10.4f}" for arm in ARMS))


def interval(q, a, b):
    return telecom.block_interval(results[q][a], results[q][b], results[q]["blocks"])


print("\nThe bar (95% intervals over spatial blocks):")
gains, ties, clearly_worse = 0, 0, 0
for q in results:
    p_vs_r, p_vs_h = interval(q, "R+P", "R"), interval(q, "R+P", "R+H")
    gains += p_vs_r[2] < 0
    ties += p_vs_h[2] < 0.01
    clearly_worse += p_vs_h[1] > 0.01
    print(f"  {q:<18} R+P vs R {p_vs_r[0]:+.1%} [{p_vs_r[1]:+.1%}, {p_vs_r[2]:+.1%}] (needs < 0)   "
          f"R+P vs R+H {p_vs_h[0]:+.1%} [{p_vs_h[1]:+.1%}, {p_vs_h[2]:+.1%}] (needs < +1%)")
print(f"  (1) profile words beat raw alone on {gains} of 4 (needs 3); "
      f"(2) within the margin of hand-built on {ties} of 4 (needs 3)")
if gains >= 3 and ties >= 3:
    verdict = "PASS: a learned profile replaces the hand-built feature"
elif gains >= 3:
    verdict = "(1) WITHOUT (2): the profile words add the square's context, but less than the hand-built feature"
elif clearly_worse >= 2:
    verdict = "FAIL: the profile words don't replace the hand-built feature"
else:
    verdict = "TOO CLOSE TO CALL: not shown; no reruns"
print(f"Verdict: {verdict}")

print("\nReported, no bar:")
for q in results:
    h_vs_r, all_vs_h = interval(q, "R+H", "R"), interval(q, "R+H+P", "R+H")
    print(f"  {q:<18} R+H vs R {h_vs_r[0]:+.1%} [{h_vs_r[1]:+.1%}, {h_vs_r[2]:+.1%}]   "
          f"R+H+P vs R+H {all_vs_h[0]:+.1%} [{all_vs_h[1]:+.1%}, {all_vs_h[2]:+.1%}]")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
