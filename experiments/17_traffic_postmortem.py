"""Experiment 17: PEMS-BAY post-mortem. Can any reasonable forecaster get a real gain from neighbours?

In experiments 12-13 one shared LightGBM got only ~2.5% from neighbours' raw
data, while the published DCRNN, which uses the road network, is far better.
Is the forecaster the bottleneck, or do neighbours not help much here?
Raw data only, no words. Three forecaster types, each trained WITHOUT and
WITH neighbours, everything else identical:

    shared LightGBM     one model for all sensors (as in experiments 12-13);
                        with = + the 5 nearest sensors' last hour
    linear per sensor   one ridge regression per sensor, so each sensor learns
                        its own weights for its own 5 nearest neighbours
    graph network       a mini DCRNN (fam/traffic_graph.py): messages passed along
                        all road links, two hops, both directions; without = the
                        same network with message passing switched off

60-minute forecast on the test months. Gain = 1 - error with / error without,
with a 95% interval from resampling whole test days.

Bar, set by the developer before any results (README):
    network helps: the same forecaster with neighbours cuts the 60-min error by
                   >= 5%, and the whole 95% interval clears 5%
    fail: no reasonable forecaster finds a real neighbour gain, so "many
          connected points" is revisited
    too close to call (interval crosses 5%) counts as not shown; no reruns with tweaks
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import lightgbm as lgb
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge

from fam import traffic_graph
from fam.traffic import (HISTORY, day_bootstrap_gain, load_speeds, nearest_neighbours, road_weights,
                         split_points)

STEPS = 12              # 60 minutes ahead
N_NEIGHBOURS = 5
LGBM_EVERY, OTHERS_EVERY, TEST_EVERY = 4, 2, 3  # time-step sampling: train (LightGBM / others), test
start_all = time.time()

speeds, times, sensors = load_speeds()
n_steps, n_sensors = speeds.shape
val_start, test_start = split_points(n_steps)
windows = sliding_window_view(speeds, HISTORY, axis=0)  # windows[t - 11] = readings t-11 .. t
neighbours, _ = nearest_neighbours(sensors, N_NEIGHBOURS, "any")
weights = road_weights(sensors)


def ends_between(start, stop, every=1):
    """Time steps t (last reading of the input hour) whose forecast lands before `stop`."""
    return np.arange(max(start, HISTORY - 1), stop - STEPS, every)


test_ends = ends_between(test_start, n_steps, TEST_EVERY)
actual = speeds[test_ends + STEPS]                     # (test times, sensors)
known = actual > 0                                     # 0 = missing reading
days = np.repeat(times[test_ends + STEPS].normalize().values[:, None], n_sensors, axis=1)[known]
hour = (times.hour + times.minute / 60).values
weekend = (times.dayofweek >= 5).astype(np.float32)
errors = {}  # (forecaster, "without"/"with") -> absolute errors on the known test points


def record(name, variant, predicted):
    errors[name, variant] = np.abs(predicted - actual)[known]
    print(f"  {name}, {variant}: {errors[name, variant].mean():.3f} mph ({time.time() - start_all:.0f} s since start)")


# 1. Shared LightGBM (experiments 12-13).
print("Shared LightGBM")
train_ends = ends_between(0, val_start, LGBM_EVERY)


def lgbm_rows(ends, with_neighbours):
    lags = windows[ends - (HISTORY - 1)]                              # (times, sensors, 12)
    cols = [lags.reshape(-1, HISTORY),
            np.repeat(hour[ends], n_sensors)[:, None], np.repeat(times[ends].dayofweek.values, n_sensors)[:, None]]
    if with_neighbours:
        cols.append(lags[:, neighbours].reshape(len(ends) * n_sensors, -1))
    return np.hstack(cols).astype(np.float32)


for variant in ("without", "with"):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63,
                              n_jobs=CPU_THREADS, random_state=0, verbose=-1)
    model.fit(lgbm_rows(train_ends, variant == "with"), speeds[train_ends + STEPS].reshape(-1))
    record("shared LightGBM", variant,
           model.predict(lgbm_rows(test_ends, variant == "with")).reshape(len(test_ends), n_sensors))

# 2. One linear (ridge) model per sensor: each sensor learns its own neighbour weights.
print("Linear per sensor")
train_ends = ends_between(0, val_start, OTHERS_EVERY)
hour_onehot = np.eye(24, dtype=np.float32)[times.hour.values]


def ridge_rows(ends, sensor, with_neighbours):
    cols = [windows[ends - (HISTORY - 1), sensor], hour_onehot[ends], weekend[ends][:, None]]
    if with_neighbours:
        cols.append(windows[ends - (HISTORY - 1)][:, neighbours[sensor]].reshape(len(ends), -1))
    return np.hstack(cols)


for variant in ("without", "with"):
    predicted = np.zeros_like(actual)
    for sensor in range(n_sensors):
        model = Ridge(alpha=1.0).fit(ridge_rows(train_ends, sensor, variant == "with"), speeds[train_ends + STEPS, sensor])
        predicted[:, sensor] = model.predict(ridge_rows(test_ends, sensor, variant == "with"))
    record("linear per sensor", variant, predicted)

# 3. Graph network (mini DCRNN), with message passing switched off for "without".
print("Graph network")
time_features = np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), weekend])
val_ends = ends_between(val_start, test_start, 6)
for variant in ("without", "with"):
    model = traffic_graph.train(speeds, time_features, weights, train_ends, val_ends, STEPS, use_graph=variant == "with")
    record("graph network", variant, traffic_graph.predict(model, speeds, time_features, test_ends, STEPS))

# Verdict against the developer's bar.
print("\n60-minute error on the test months (mph), and the gain from neighbours with its 95% interval (by day):\n")
print(f"{'forecaster':<20}{'without':>9}{'with':>9}{'gain':>8}   {'95% interval':<18}verdict")
passed = False
for name in ("shared LightGBM", "linear per sensor", "graph network"):
    gain, low, high = day_bootstrap_gain(errors[name, "without"], errors[name, "with"], days)
    verdict = "CLEAR PASS" if low >= 0.05 else "CLEAR FAIL" if high < 0.05 else "TOO CLOSE TO CALL (not shown)"
    passed |= low >= 0.05
    print(f"{name:<20}{errors[name, 'without'].mean():>9.3f}{errors[name, 'with'].mean():>9.3f}{gain:>8.1%}   "
          f"[{low:+.1%}, {high:+.1%}]   {verdict}")
best = min(errors, key=lambda k: errors[k].mean())
print(f"\nBest forecaster: {best[0]} {best[1]} neighbours, {errors[best].mean():.3f} mph (published DCRNN: 2.07)")
print(f"Network helps (at least one forecaster clears 5% with its whole interval): {'PASS' if passed else 'FAIL'}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
