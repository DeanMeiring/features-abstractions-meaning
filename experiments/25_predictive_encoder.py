"""Experiment 25: does a predictive encoder remove the squeeze? (fix 1 from the audit)

Experiment 24 found the encoder, trained only to rebuild the past day, throws away what
next-hour forecasting needs: a ceiling no number of words can pass. Here the one change:
the encoder and its correction words are trained to rebuild the past 24 hours AND predict
the next 6 hours (training weeks only), weighted equally. Everything else is experiment 22b.
At test time the words still come from the past 24 hours alone.

    1. train the dictionary (revive off, learning rate 0.0005 -> 0, best pass kept)
    2. health gate: last pass within 10% of the best (combined check error), past-part
       rebuild error at most 0.10 with 16 words, combined error falls 1 -> 4 -> 16 words;
       if it fails, the run is invalid and stops here (stop rule)
    3. experiment 21's forecasting bar, unchanged, on the test week:
       at 16 words, internet and calls each within +10% of raw data (whole 95% interval
       over spatial blocks), and 16 words better than 1 word on both
    4. reported: the audit's three checks on the new dictionary
Bar committed in the README before this was built.
    python experiments/25_predictive_encoder.py --smoke   # all 6 passes, small, 20 trees (not a result)
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
MODEL_FILE = ROOT / "data" / "models" / ("telecom_predictive_words_smoke.pt" if SMOKE else "telecom_predictive_words.pt")
HISTORY, AHEAD, KS = 24, 6, [1, 2, 4, 8, 16]
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))
spread = logs[:, :train_hours].std(axis=(0, 1))
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)
train_t = np.arange(HISTORY - 1, train_hours - 1)          # forecasting examples: t + 1 in the training weeks
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)     # t + 1 in the test week


def history(t_values):
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h, hours=HISTORY):
    return ((h.reshape(len(h), hours, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


def next_hours(t_values):
    """The 6 hours after each t, all three activities: (squares * len(t), 18), standardised."""
    ahead = np.stack([logs[:, t_values + j] for j in range(1, AHEAD + 1)], axis=2)   # (squares, t, 6, 3)
    return standardise(ahead.reshape(-1, AHEAD * 3), AHEAD)


# Step 1: train the predictive dictionary on training-week windows whose next 6 hours are also in the training weeks.
dict_t = np.arange(HISTORY - 1, train_hours - AHEAD)
past_all, future_all = history(dict_t), next_hours(dict_t)
size = min(1_000_000, len(past_all) - 20_000)
fit_rows = np.random.default_rng(0).choice(len(past_all), size=size, replace=False)
check_rows = np.random.default_rng(1).choice(np.setdiff1d(np.arange(len(past_all)), fit_rows), size=20_000, replace=False)
fit_x, fit_f = standardise(past_all[fit_rows]), future_all[fit_rows]
check_x, check_f = standardise(past_all[check_rows]), future_all[check_rows]
del past_all, future_all
print(f"Training the predictive dictionary on {len(fit_x):,} windows (rebuild the past 24 h + predict the next {AHEAD} h)...")
t = time.time()
model = residual_words.train(fit_x, epochs=6, lr=5e-4, check=check_x, lr_to_zero=True, keep_best=True, revive=False,
                             future=fit_f, check_future=check_f)
print(f"  took {time.time() - t:.0f} s")
torch.save(model.state_dict(), MODEL_FILE)

# Step 2: the health gate.
history_16 = model.history[:, 15]
best_pass = int(history_16.argmin())
combined = residual_words.rebuild_errors(model, check_x, check_f)
past_only = residual_words.rebuild_errors(model, check_x)
gate = {
    "(1) last pass within 10% of the best (combined)": history_16[-1] <= 1.10 * history_16[best_pass],
    "(2) past-part 16-word rebuild error <= 0.10": past_only[15] <= 0.10,
    "(3) combined error falls 1 -> 4 -> 16 words": combined[0] > combined[3] > combined[15],
}
print(f"\nCombined 16-word check error, pass by pass: " + " / ".join(f"{e:.3f}" for e in history_16))
print(f"Kept pass {best_pass + 1}: combined 1 / 4 / 16 words {combined[0]:.3f} / {combined[3]:.3f} / {combined[15]:.3f}; "
      f"past part {past_only[0]:.3f} / {past_only[3]:.3f} / {past_only[15]:.3f}")
print("Health gate:")
for check, ok in gate.items():
    print(f"  {check}: {'yes' if ok else 'NO'}")
if not all(gate.values()):
    if not SMOKE:
        sys.exit("  -> FAIL: the run is invalid; no forecasting (stop rule)")
    print("  -> FAIL (smoke run only: continuing anyway, just to test the forecasting code)")
else:
    print("  -> HEALTHY")

# Step 3: experiment 21's forecasting test, unchanged, on the test week.
books = model.codebooks.detach().numpy()


@torch.no_grad()
def encode(t_values):
    zs, ws = [], []
    for i in range(0, len(t_values), 24):
        z = model.encoder(torch.tensor(standardise(history(t_values[i:i + 24]))))
        zs.append(z.numpy().reshape(n_sq, -1, 32))
        ws.append(model.quantize(z)[0].numpy().astype(np.uint8).reshape(n_sq, -1, 16))
    return np.concatenate(zs, axis=1).reshape(-1, 32), np.concatenate(ws, axis=1).reshape(-1, 16)


def extras(t_values):
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


def prefix(words, k):
    return sum(books[j][words[:, j]] for j in range(k)).astype(np.float32)


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
z_train, w_train = encode(train_t)
z_test, w_test = encode(test_t)
print(f"Words for every square-hour ready ({time.time() - start_all:.0f} s since start)")


def fit_and_test(X_train, X_test):
    out = {}
    for q in train_y:
        m = (lgb.LGBMClassifier if q == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS).fit(X_train, train_y[q])
        out[q] = errors(q, m.predict_proba(X_test)[:, 1] if q == "Q3 unusual" else m.predict(X_test), test_y[q])
    return out


results = {"raw": fit_and_test(np.hstack([history(train_t).astype(np.float32), extras(train_t)]),
                               np.hstack([history(test_t).astype(np.float32), extras(test_t)]))}
results["encoder numbers"] = fit_and_test(np.hstack([z_train, extras(train_t)]), np.hstack([z_test, extras(test_t)]))
for k in KS:
    results[k] = fit_and_test(np.hstack([prefix(w_train, k), extras(train_t)]), np.hstack([prefix(w_test, k), extras(test_t)]))
    print(f"  words, k = {k} done ({time.time() - start_all:.0f} s since start)")

arms = ["raw", "encoder numbers"] + KS
print("\nTest-week error (Q1, Q2: mean absolute error of log(1 + x); Q3: log loss):\n")
print(f"{'':<14}" + "".join(f"{('k=' + str(a)) if isinstance(a, int) else a:>16}" for a in arms))
for q in train_y:
    print(f"{q:<14}" + "".join(f"{results[a][q].mean():>16.4f}" for a in arms))

print("\nVs raw data (relative error, 95% interval over spatial blocks):")
verdicts = []
for q in ("Q1 internet", "Q2 calls"):
    enc = telecom.block_interval(results["encoder numbers"][q], results["raw"][q], test_blocks)
    gap = telecom.block_interval(results[16][q], results["raw"][q], test_blocks)
    better = telecom.block_interval(results[16][q], results[1][q], test_blocks)
    verdicts.append((q, gap, better))
    print(f"  {q}: encoder numbers {enc[0]:+.1%} [{enc[1]:+.1%}, {enc[2]:+.1%}] | 16 words {gap[0]:+.1%} "
          f"[{gap[1]:+.1%}, {gap[2]:+.1%}] | 16 words vs 1 word {better[0]:+.1%} [{better[1]:+.1%}, {better[2]:+.1%}]")
passes = all(gap[2] < 0.10 and better[2] < 0 for _, gap, better in verdicts)
fails = any(gap[1] > 0.10 for _, gap, _ in verdicts)
print("Verdict (experiment 21's bar): " + ("PASS: the predictive words close the gap" if passes else
      "FAIL: the gap stays" if fails else "TOO CLOSE TO CALL: not shown; no reruns"))

# Step 4: the audit's checks on the new dictionary (reported, no bar).
bits = []
for k in range(16):
    p = np.bincount(w_test[:, k], minlength=256) / len(w_test)
    p = p[p > 0]
    bits.append(float(-(p * np.log2(p)).sum()))
sample = np.random.default_rng(0).choice(len(w_test), size=min(200_000, len(w_test)), replace=False)
x_sample = standardise(history(test_t)[sample])
with torch.no_grad():
    rebuilt = model.decoder(torch.tensor(prefix(w_test[sample], 16))).numpy()[:, :HISTORY * 3]
per_hour = ((rebuilt - x_sample) ** 2).reshape(-1, HISTORY, 3).mean(axis=(0, 2))
print(f"\nAudit A, bits per word position: " + " ".join(f"{b:.1f}" for b in bits) + f"  (together {sum(bits):.1f} of 128)")
print(f"Audit B, last hour's rebuild error vs the 24-hour average: {per_hour[-1]:.3f} vs {per_hour.mean():.3f}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
