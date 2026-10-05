"""Experiment 11: the raw-data baseline on PEMS-BAY road traffic.

Task (the standard benchmark): from a sensor's last hour of speeds (12
readings, 5 minutes apart), predict its speed 15, 30 and 60 minutes ahead.
Train on the first 70% of the six months, test on the last 20% (the future).

Two baselines, both reading raw data only:
    last value  predict that the speed stays what it is now
    LightGBM    one shared model for all 325 sensors, from the last 12
                readings plus time of day and day of week

Records error (MAE, mph), training time, prediction time and input size, so
later experiments can show "same error, smaller and faster" against this.
Sanity check: published results (DCRNN paper, Li et al. 2018) put simple
methods around 2-3 mph and DCRNN at about 1.38 / 1.74 / 2.07 mph.
"""

import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import lightgbm as lgb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from fam.traffic import HISTORY, HORIZONS, load_speeds, mae, split_points

TRAIN_EVERY = 4  # use every 4th training time step: 4x less training data, keeps the run to minutes

speeds, times, sensors = load_speeds()
n_steps, n_sensors = speeds.shape
val_start, test_start = split_points(n_steps)
longest = max(HORIZONS.values())

# Every window of 12 readings: windows[t] = readings t-11 .. t, for every sensor.
windows = sliding_window_view(speeds, HISTORY, axis=0)  # (time - 11, sensors, 12)


def window_ends(start, stop, every=1):
    """Time steps t (last reading of the input hour) whose every forecast lands before `stop`."""
    return np.arange(max(start, HISTORY - 1), stop - longest, every)


def rows(ends):
    """One row per (time step, sensor): 12 readings + time of day + day of week."""
    lags = windows[ends - (HISTORY - 1)].reshape(-1, HISTORY)
    hour = np.repeat((times[ends].hour + times[ends].minute / 60).values, n_sensors)
    weekday = np.repeat(times[ends].dayofweek.values, n_sensors)
    return np.column_stack([lags, hour, weekday]).astype(np.float32)


def targets(ends, steps):
    return speeds[ends + steps].reshape(-1)


train_ends = window_ends(0, val_start, TRAIN_EVERY)
test_ends = window_ends(test_start, n_steps)
X_train, X_test = rows(train_ends), rows(test_ends)
print(f"Training rows: {len(X_train):,}   test rows: {len(X_test):,}   inputs per row: {X_train.shape[1]}")

raw_bytes = speeds.astype(np.float32).tobytes()
print(f"Raw data: {len(raw_bytes) / 1e6:.0f} MB as 4-byte numbers, "
      f"{len(gzip.compress(raw_bytes, compresslevel=6)) / 1e6:.0f} MB zipped "
      f"({n_steps:,} steps x {n_sensors} sensors)\n")

print(f"{'horizon':>8}  {'last value':>10}  {'LightGBM':>9}  {'train s':>8}  {'predict s':>9}")
for name, steps in HORIZONS.items():
    actual = targets(test_ends, steps)
    persistence = mae(X_test[:, HISTORY - 1], actual)

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63,
                              n_jobs=CPU_THREADS, random_state=0, verbose=-1)
    t = time.time()
    model.fit(X_train, targets(train_ends, steps))
    train_s = time.time() - t
    t = time.time()
    predicted = model.predict(X_test)
    predict_s = time.time() - t
    print(f"{name:>8}  {persistence:>10.2f}  {mae(predicted, actual):>9.2f}  {train_s:>8.0f}  {predict_s:>9.1f}")

print("\nPublished for reference (DCRNN paper): DCRNN 1.38 / 1.74 / 2.07 mph at 15 / 30 / 60 min.")
