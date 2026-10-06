"""Experiment 20: the telecom reuse test, the project's one-shot test (option C, "less data").

One dictionary, learned once from the training weeks' raw activity (no future hours,
no answers), describes each Milan grid square's last 24 hours as 8 words. Those same
words are reused for 3 different questions about the next hour:
    Q1 internet    log(1 + internet), mean absolute error
    Q2 calls       log(1 + calls), mean absolute error
    Q3 unusual     is internet more than double or less than half the square's typical
                   value for that hour (weekday/weekend)? log loss
Same LightGBM for every arm. The claim: words trained on a fixed random share of the
training examples (2%, set by the rule in experiment 20a) get within 2% of raw data
trained on 100%.

Bar (README, committed before running; full definitions there):
    pass:        on >= 2 of 3 questions, words@2% / raw@100% - 1 has its whole 95% interval
                 below +2%, AND (safeguard) raw@2% / raw@100% - 1 has its whole interval above +2%
    clear fail:  on >= 2 of 3 questions, the whole interval is above +2% -> rethink the claim
    otherwise:   too close to call, counts as not shown; no reruns with tweaks
Intervals resample the 100 spatial blocks of 10 x 10 squares.
    python experiments/20_telecom_reuse.py --smoke   # tiny run to check the code (not a result)
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

from fam import telecom, window_words

SMOKE = "--smoke" in sys.argv
PART = "2%"
SHARE = 0.02  # set by the rule in experiment 20a (committed before the official run); was 10% in the first bar
MODEL_FILE = ROOT / "data" / "models" / ("telecom_words_smoke.pt" if SMOKE else "telecom_words.pt")
HISTORY = 24
SETTINGS = dict(n_estimators=20 if SMOKE else 300, learning_rate=0.1, num_leaves=63, random_state=0,
                n_jobs=CPU_THREADS, verbose=-1)
start_all = time.time()

# Step 1: hourly activity, log scale. Squares x hours x (sms, calls, internet).
counts = telecom.load_hourly()
squares = np.arange(400) if SMOKE else np.arange(telecom.SQUARES)
logs = np.log1p(counts[squares])
blocks_of_square = telecom.spatial_blocks()[squares]
n_sq = len(squares)
train_hours = telecom.TRAIN_HOURS
mean = logs[:, :train_hours].mean(axis=(0, 1))   # training-period statistics only
spread = logs[:, :train_hours].std(axis=(0, 1))

# Each history window: hours t-23..t, flattened to 72 numbers (hour-major: 24 x 3).
windows = sliding_window_view(logs, HISTORY, axis=1).transpose(0, 1, 3, 2)  # (squares, t-23, 24, 3)
train_t = np.arange(HISTORY - 1, train_hours - 1)                 # t + 1 still in the training weeks
test_t = np.arange(train_hours - 1, telecom.HOURS - 1)            # t + 1 in the test week


def history(t_values):
    """(squares * len(t), 72) log-scale histories for hours t, square-major."""
    return windows[:, t_values - (HISTORY - 1)].reshape(n_sq * len(t_values), HISTORY * 3)


def standardise(h):
    return ((h.reshape(len(h), HISTORY, 3) - mean) / spread).reshape(len(h), -1).astype(np.float32)


# Step 2: the dictionary, learned once from training-period histories only (rebuild only).
if MODEL_FILE.exists():
    words_model = window_words.WindowWords()
    words_model.load_state_dict(torch.load(MODEL_FILE))
    words_model.eval()
    print(f"Loaded {MODEL_FILE.name}")
else:
    rng = np.random.default_rng(0)
    all_train = history(train_t)
    sample = all_train[rng.choice(len(all_train), size=min(1_000_000, len(all_train)), replace=False)]
    print(f"Training the dictionary on {len(sample):,} training windows (rebuild only)...")
    t = time.time()
    words_model = window_words.train(standardise(sample), epochs=1 if SMOKE else 6)
    print(f"  took {time.time() - t:.0f} s")
    torch.save(words_model.state_dict(), MODEL_FILE)
    del all_train, sample
with torch.no_grad():
    table = words_model.codebook.numpy()


# Step 3: inputs for both sides, with the same extras (hour of day, weekend flag).
def extras(t_values):
    hour = (t_values % 24).astype(np.float32)
    weekend = ((t_values // 24) % 7 >= 5).astype(np.float32)   # day 0 is a Monday
    return np.tile(np.column_stack([hour, weekend]), (n_sq, 1))


def raw_inputs(t_values):
    return np.hstack([history(t_values).astype(np.float32), extras(t_values)])


def word_inputs(t_values):
    """Words for every (square, t), translated a day of hours at a time to keep memory low."""
    words = np.zeros((n_sq, len(t_values), 8), dtype=np.uint8)
    for i in range(0, len(t_values), 24):
        chunk = t_values[i:i + 24]
        words[:, i:i + len(chunk)] = window_words.words_of(words_model, standardise(history(chunk))).reshape(n_sq, len(chunk), 8)
    words = np.sort(words.reshape(-1, 8), axis=1)
    return np.hstack([table[words].reshape(len(words), -1), extras(t_values)]), words


# Step 4: the three questions' answers, for hour t + 1.
weekday_days = [d for d in range(14) if d % 7 < 5]
weekend_days = [d for d in range(14) if d % 7 >= 5]
internet = logs[:, :, 2]
by_day = internet[:, :train_hours].reshape(n_sq, 14, 24)
typical = np.stack([np.median(by_day[:, weekday_days], axis=1), np.median(by_day[:, weekend_days], axis=1)], axis=2)


def answers(t_values):
    nxt = t_values + 1
    q1 = internet[:, nxt].reshape(-1)
    q2 = logs[:, nxt, 1].reshape(-1)
    kind = ((nxt // 24) % 7 >= 5).astype(int)
    q3 = (np.abs(internet[:, nxt] - typical[:, nxt % 24, kind]) > np.log(2)).astype(int).reshape(-1)
    return {"Q1 internet": q1, "Q2 calls": q2, "Q3 unusual": q3}


train_y, test_y = answers(train_t), answers(test_t)
print(f"Examples: {len(train_t) * n_sq:,} training, {len(test_t) * n_sq:,} test. "
      f"Unusual flag: {train_y['Q3 unusual'].mean():.1%} of training, {test_y['Q3 unusual'].mean():.1%} of test hours")
few = np.sort(np.random.default_rng(0).choice(len(train_t) * n_sq, size=int(len(train_t) * n_sq * SHARE), replace=False))
test_blocks = np.repeat(blocks_of_square, len(test_t))


def errors(question, predicted, actual):
    if question == "Q3 unusual":
        p = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -(actual * np.log(p) + (1 - actual) * np.log(1 - p))
    return np.abs(predicted - actual)


# Step 5: the four arms, every question, identical model settings.
results, seconds = {}, {}
for name in ("raw", "words"):
    t = time.time()
    if name == "raw":
        X_train, X_test = raw_inputs(train_t), raw_inputs(test_t)
    else:
        (X_train, w_train), (X_test, w_test) = word_inputs(train_t), word_inputs(test_t)
        first_day = test_t[:24]  # rebuild error on the test week's first day of histories
        w_day = window_words.words_of(words_model, standardise(history(first_day)))
        with torch.no_grad():
            rebuilt = words_model.draw(torch.tensor(table[w_day])).numpy()
        rebuild_error = float(((rebuilt - standardise(history(first_day))) ** 2).mean())
    print(f"{name} inputs ready: {X_train.shape[1]} numbers per example ({time.time() - t:.0f} s)")
    for share, rows in (("100%", slice(None)), (PART, few)):
        for question in train_y:
            model = (lgb.LGBMClassifier if question == "Q3 unusual" else lgb.LGBMRegressor)(**SETTINGS)
            t = time.time()
            model.fit(X_train[rows], train_y[question][rows])
            seconds[name, share, question] = time.time() - t
            predicted = model.predict_proba(X_test)[:, 1] if question == "Q3 unusual" else model.predict(X_test)
            results[name, share, question] = errors(question, predicted, test_y[question])
        print(f"  {name} @ {share} done ({time.time() - start_all:.0f} s since start)")
    del X_train, X_test

# Step 6: verdict, against the bar fixed before running.
print(f"\nTest-week error (Q1, Q2: mean absolute error of log(1 + x); Q3: log loss). Dictionary rebuild error "
      f"{rebuild_error:.3f} (standardised units)\n")
print(f"{'question':<13}{'raw 100%':>10}{'raw ' + PART:>10}{'words ' + PART:>11}{'words 100%':>12}")
for q in train_y:
    print(f"{q:<13}" + "".join(f"{results[arm + (q,)].mean():>{w}.4f}"
                               for arm, w in ((("raw", "100%"), 10), (("raw", PART), 10), (("words", PART), 11),
                                              (("words", "100%"), 12))))

passes = fails = 0
print("\nPer question (relative error vs raw at 100%, 95% interval over spatial blocks):")
for q in train_y:
    claim = telecom.block_interval(results["words", PART, q], results["raw", "100%", q], test_blocks)
    guard = telecom.block_interval(results["raw", PART, q], results["raw", "100%", q], test_blocks)
    option_a = telecom.block_interval(results["words", "100%", q], results["raw", "100%", q], test_blocks)
    counts_as_pass = claim[2] < 0.02 and guard[1] > 0.02
    clear_fail = claim[1] > 0.02
    passes += counts_as_pass
    fails += clear_fail
    verdict = "PASS" if counts_as_pass else "CLEAR FAIL" if clear_fail else (
        "within 2% but safeguard not met" if claim[2] < 0.02 else "too close to call")
    print(f"  {q}: words@{PART} {claim[0]:+.1%} [{claim[1]:+.1%}, {claim[2]:+.1%}] | safeguard raw@{PART} "
          f"{guard[0]:+.1%} [{guard[1]:+.1%}, {guard[2]:+.1%}] | option A words@100% {option_a[0]:+.1%} "
          f"[{option_a[1]:+.1%}, {option_a[2]:+.1%}]  -> {verdict}")

print("\nOption B (reported): bytes per example raw 72 numbers (288 as float32) vs words 8 bytes; training seconds:")
for q in train_y:
    print(f"  {q}: " + ", ".join(f"{n}@{s} {seconds[n, s, q]:.0f}s" for n in ("raw", "words") for s in ("100%", PART)))
overall = "PASS (proven, per the project bar)" if passes >= 2 else "CLEAR FAIL (rethink the claim)" if fails >= 2 \
    else "TOO CLOSE TO CALL (not shown; no reruns)"
print(f"\nProject bar, option C: {passes} of 3 questions pass, {fails} clearly fail -> {overall}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min" + ("   [SMOKE RUN: not a result]" if SMOKE else ""))
