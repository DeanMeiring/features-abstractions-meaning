"""Experiment 13: do neighbours help once the connections have a direction?

Experiment 12 gave each sensor its 5 nearest neighbours in ANY direction, and
they barely helped. Traffic has a direction: cars flow forward, but a jam's
queue grows backwards from a bottleneck ahead. So here each sensor gets its 5
nearest neighbours either AHEAD (downstream) or BEHIND (upstream), from the
directional road distances. Same LightGBM and the same hour words as
experiment 12 (reused, not retrained), so only the choice of neighbours changes.

    A          own last hour (+ time of day, day of week)
    B ahead    A + 5 neighbours ahead, raw last hours   (60 bytes)
    B behind   A + 5 neighbours behind, raw last hours  (60 bytes)
    C ahead    A + 5 neighbours ahead, 2 words each     (10 bytes)
    C behind   A + 5 neighbours behind, 2 words each    (10 bytes)

Success bar, fixed in the README before running, on the 60-minute forecast:
    direction matters:    the better direction's B cuts A's error by >= 3%
    words keep the gain:  that direction's C keeps >= half of B's improvement over A,
                          with neighbours sending >= 5x fewer bytes
Needs experiment 12 first (for the hour-words model).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # save pictures to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from fam import traffic_words
from fam.traffic import HISTORY, HORIZONS, load_speeds, mae, nearest_neighbours, split_points

TRAIN_EVERY, TEST_EVERY = 4, 3  # same sampling as experiment 12
N_NEIGHBOURS = 5
PICTURE = ROOT / "results" / "13_directional_forecasts.png"
start_all = time.time()

speeds, times, sensors = load_speeds()
n_steps, n_sensors = speeds.shape
val_start, test_start = split_points(n_steps)
longest = max(HORIZONS.values())
windows = sliding_window_view(speeds, HISTORY, axis=0)  # windows[t - 11] = readings t-11 .. t

# Step 1: neighbours ahead and behind.
neighbours = {}
for direction in ("ahead", "behind"):
    neighbours[direction], filled = nearest_neighbours(sensors, N_NEIGHBOURS, direction)
    print(f"Neighbours {direction}: {filled} sensors have fewer than {N_NEIGHBOURS} road links that way "
          f"(gaps filled by map distance)")

# Step 2: every sensor's words for every hour, from experiment 12's dictionary.
words_model = traffic_words.HourWords()
words_model.load_state_dict(torch.load(ROOT / "data" / "models" / "pems_hour_words.pt"))
words_model.eval()
words_all = np.concatenate([
    traffic_words.words_of(words_model, windows[i:i + 2000].reshape(-1, HISTORY)).reshape(-1, n_sensors, 2)
    for i in range(0, len(windows), 2000)
])


def ends_between(start, stop, every=1):
    """Time steps t (last reading of the input hour) whose every forecast lands before `stop`."""
    return np.arange(max(start, HISTORY - 1), stop - longest, every)


def own(ends):
    lags = windows[ends - (HISTORY - 1)].reshape(-1, HISTORY)
    hour = np.repeat((times[ends].hour + times[ends].minute / 60).values, n_sensors)
    weekday = np.repeat(times[ends].dayofweek.values, n_sensors)
    return np.column_stack([lags, hour, weekday]).astype(np.float32)


def raw_of(direction):
    return lambda e: np.column_stack(
        [own(e), windows[e - (HISTORY - 1)][:, neighbours[direction]].reshape(len(e) * n_sensors, -1)])


def words_of(direction):
    return lambda e: np.column_stack(
        [own(e), words_all[e - (HISTORY - 1)][:, neighbours[direction]].reshape(len(e) * n_sensors, -1)
         .astype(np.float32)])


INPUTS = {
    "A alone": own,
    "B ahead, raw": raw_of("ahead"),
    "B behind, raw": raw_of("behind"),
    "C ahead, words": words_of("ahead"),
    "C behind, words": words_of("behind"),
}
WORD_COLUMNS = list(range(14, 14 + 2 * N_NEIGHBOURS))  # word numbers are categories, not amounts
BYTES = {name: 0 if name.startswith("A") else N_NEIGHBOURS * (HISTORY if "raw" in name else 2) for name in INPUTS}

train_ends = ends_between(0, val_start, TRAIN_EVERY)
test_ends = ends_between(test_start, n_steps, TEST_EVERY)

# The picture: the same day and sensor as experiment 12 (the deepest test-month jam).
dates = times.normalize()
day = pd.Timestamp("2017-06-09")
sensor = int(np.where(sensors == 400209)[0][0])
plot_ends = np.where(dates == day)[0] - 12  # each forecast is made 60 minutes before the moment it predicts

# Step 3: train and test each input at each horizon.
results, curves = {}, {}
for name, build in INPUTS.items():
    X_train, X_test, X_plot = build(train_ends), build(test_ends), build(plot_ends)
    categorical = WORD_COLUMNS if "words" in name else []
    for horizon, steps in HORIZONS.items():
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63,
                                  n_jobs=CPU_THREADS, random_state=0, verbose=-1)
        t = time.time()
        model.fit(X_train, speeds[train_ends + steps].reshape(-1), categorical_feature=categorical)
        train_s = time.time() - t
        results[name, horizon] = (mae(model.predict(X_test), speeds[test_ends + steps].reshape(-1)), train_s)
        if steps == 12:
            curves[name] = model.predict(X_plot).reshape(len(plot_ends), n_sensors)[:, sensor]
    del X_train, X_test
    print(f"{name} done ({time.time() - start_all:.0f} s since start)")

print("\nAverage error (mph) on the test months; training seconds in brackets:\n")
print(f"{'input':<17}{'bytes from neighbours':>22}" + "".join(f"{h:>15}" for h in HORIZONS))
for name in INPUTS:
    print(f"{name:<17}{BYTES[name]:>22}"
          + "".join(f"{results[name, h][0]:>9.2f} ({results[name, h][1]:>3.0f}s)" for h in HORIZONS))
print(f"{'DCRNN (published)':<17}{'':>22}" + "".join(f"{v:>15.2f}" for v in (1.38, 1.74, 2.07)))

# Step 4: verdict against the bar fixed before running (60-minute forecast).
err = {name: results[name, "60 min"][0] for name in INPUTS}
best = min(("ahead", "behind"), key=lambda d: err[f"B {d}, raw"])
a, b, c = err["A alone"], err[f"B {best}, raw"], err[f"C {best}, words"]
cut = (a - b) / a
kept = (a - c) / (a - b) if a > b else float("nan")
ratio = BYTES[f"B {best}, raw"] / BYTES[f"C {best}, words"]
print(f"\nVerdict (60-minute forecast), better direction: {best}")
print(f"  Direction matters: B {best} cuts A's error by {cut:.1%} (bar: >= 3%)  {'PASS' if cut >= 0.03 else 'FAIL'}")
print(f"  Words keep the gain: C {best} keeps {kept:.0%} of B's improvement (bar: >= 50%), "
      f"{ratio:.0f}x fewer bytes (bar: >= 5x)  {'PASS' if kept >= 0.5 and ratio >= 5 else 'FAIL'}")

# Step 5: picture. Left: the jam, actual vs 60-min forecasts. Right: 60-min error per input.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
lines = {"A alone": "#2a78d6", f"B {best}, raw": "#eb6834", f"C {best}, words": "#1baf7a"}
target = times[plot_ends + 12]
hours = target.hour + target.minute / 60
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.8, 1]}, facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)

left.grid(axis="y", color=GRID, linewidth=0.8)
left.plot(hours, speeds[plot_ends + 12, sensor], color=INK, linewidth=2, label="actual speed")
for name, colour in lines.items():
    left.plot(hours, curves[name], color=colour, linewidth=2, label=f"{name} (forecast 60 min earlier)")
left.set_xlim(hours[0], hours[-1])
left.set_xticks(range(0, 25, 3))
left.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
left.set_ylabel("speed (mph)", color=MUTED, fontsize=9)
left.set_title(f"Sensor {sensors[sensor]}, {day:%a %d %b %Y}: actual speed vs forecasts made 60 min earlier",
               loc="left", fontsize=10, color=INK)
left.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)

names = list(INPUTS) + ["DCRNN (published)"]
values = [err[n] for n in INPUTS] + [2.07]
right.grid(axis="x", color=GRID, linewidth=0.8)
right.barh(range(len(names))[::-1], values, height=0.6,
           color=["#2a78d6"] * len(INPUTS) + ["#a8a7a2"])
for y, v in zip(range(len(names))[::-1], values):
    right.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
right.set_yticks(range(len(names))[::-1])
right.set_yticklabels(names, fontsize=8, color=INK)
right.set_xlim(0, max(values) * 1.15)
right.set_xlabel("average error at 60 min (mph), lower is better", color=MUTED, fontsize=9)
right.set_title("60-minute error on the test months", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(PICTURE, dpi=130, facecolor=SURFACE)
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
