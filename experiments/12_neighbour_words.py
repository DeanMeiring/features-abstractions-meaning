"""Experiment 12: do neighbours' words help a road sensor forecast its speed?

A jam an hour from now usually shows up first on a neighbouring road. Each
sensor's last hour becomes 2 self-contained words (fam/traffic_words.py,
trained on the training months only). Same LightGBM, three inputs:

    A  alone           own last hour (12 readings) + time of day + day of week
    B  network, raw    A + the 5 nearest neighbours' raw last hours (60 readings)
    C  network, words  A + the 5 nearest neighbours' words (10 words = 10 bytes)

Neighbours: the 5 closest sensors by road distance (map distance fills the
gaps for the few sensors with fewer road links). Words go to LightGBM as
word numbers (categories), the way a receiver would get them.

Success bar, fixed in the README before running, on the 60-minute forecast:
    network helps:   B or C cuts A's error by at least 3%
    words keep it:   C within 2% of B, with neighbours sending >= 5x fewer bytes
Also saves a picture: one rush-hour day, actual speed vs each 60-minute forecast.
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
from fam.traffic import HISTORY, HORIZONS, load_road_links, load_speeds, mae, split_points

TRAIN_EVERY, TEST_EVERY = 4, 3  # use every 4th training / 3rd test time step, to fit in minutes and memory
N_NEIGHBOURS = 5
MODEL_FILE = ROOT / "data" / "models" / "pems_hour_words.pt"
PICTURE = ROOT / "results" / "12_forecasts.png"
start_all = time.time()

speeds, times, sensors = load_speeds()
n_steps, n_sensors = speeds.shape
val_start, test_start = split_points(n_steps)
longest = max(HORIZONS.values())
windows = sliding_window_view(speeds, HISTORY, axis=0)  # windows[t - 11] = readings t-11 .. t

# Step 1: each sensor's 5 nearest neighbours by road distance (either direction).
links = load_road_links(sensors)
index = {s: i for i, s in enumerate(sensors)}
distance = np.full((n_sensors, n_sensors), np.inf)
for a, b, metres in links[["from", "to", "metres"]].itertuples(index=False):
    i, j = index[a], index[b]
    distance[i, j] = distance[j, i] = min(distance[i, j], metres)
np.fill_diagonal(distance, np.inf)
few_links = int((np.isfinite(distance).sum(axis=1) < N_NEIGHBOURS).sum())
# A few sensors have fewer than 5 road links; fill their gaps with the closest sensors on the map
# (ranked after every road-linked one, so road links always come first).
meta = pd.read_csv(ROOT / "data" / "raw" / "pems_bay" / "pems_bay_meta.csv").set_index("sensor_id").loc[sensors]
lat, lon = np.radians(meta["Latitude"].values), np.radians(meta["Longitude"].values)
on_map = 6_371_000 * np.hypot(lat[:, None] - lat[None, :], (lon[:, None] - lon[None, :]) * np.cos(lat.mean()))
ranking = np.where(np.isfinite(distance), distance, 1e9 + on_map)
np.fill_diagonal(ranking, np.inf)
neighbours = np.argsort(ranking, axis=1)[:, :N_NEIGHBOURS]  # (sensors, 5)
print(f"Neighbours: median road distance to the 5 nearest {np.median(np.sort(distance, axis=1)[:, :5]) / 1000:.1f} km; "
      f"{few_links} sensors have fewer than 5 road links (gaps filled by map distance)")


def ends_between(start, stop, every=1):
    """Time steps t (last reading of the input hour) whose every forecast lands before `stop`."""
    return np.arange(max(start, HISTORY - 1), stop - longest, every)


# Step 2: the word model learns the dictionary from the training months only.
if MODEL_FILE.exists():
    words_model = traffic_words.HourWords()
    words_model.load_state_dict(torch.load(MODEL_FILE))
    print(f"Loaded {MODEL_FILE.name}")
else:
    rng = np.random.default_rng(0)
    ends = rng.choice(ends_between(0, val_start), size=4000, replace=False)  # 4,000 times x 325 sensors = 1.3M hours
    past = windows[ends - (HISTORY - 1)].reshape(-1, HISTORY)
    future = np.stack([speeds[ends + k] for k in range(1, 13)], axis=-1).reshape(-1, 12)
    print(f"Training hour words on {len(past):,} (last hour, next hour) pairs from the training months...")
    t = time.time()
    words_model = traffic_words.train(past, future)
    print(f"  took {time.time() - t:.0f} s")
    torch.save(words_model.state_dict(), MODEL_FILE)
words_model.eval()

# Every sensor's words for every hour window: words_all[t - 11, sensor] = its 2 words.
words_all = np.concatenate([
    traffic_words.words_of(words_model, windows[i:i + 2000].reshape(-1, HISTORY)).reshape(-1, n_sensors, 2)
    for i in range(0, len(windows), 2000)
])
print(f"Words for every sensor-hour: {words_all.shape[0] * n_sensors:,} hours, "
      f"{len(np.unique(words_all))}/256 words used")


# Step 3: the three inputs. One row per (time step, sensor).
def own(ends):
    lags = windows[ends - (HISTORY - 1)].reshape(-1, HISTORY)
    hour = np.repeat((times[ends].hour + times[ends].minute / 60).values, n_sensors)
    weekday = np.repeat(times[ends].dayofweek.values, n_sensors)
    return np.column_stack([lags, hour, weekday]).astype(np.float32)


def neighbours_raw(ends):
    return windows[ends - (HISTORY - 1)][:, neighbours].reshape(len(ends) * n_sensors, -1)  # 5 x 12 readings


def neighbours_words(ends):
    return words_all[ends - (HISTORY - 1)][:, neighbours].reshape(len(ends) * n_sensors, -1)  # 5 x 2 word numbers


INPUTS = {
    "A alone": lambda e: own(e),
    "B network, raw": lambda e: np.column_stack([own(e), neighbours_raw(e)]),
    "C network, words": lambda e: np.column_stack([own(e), neighbours_words(e).astype(np.float32)]),
}
WORD_COLUMNS = list(range(14, 14 + 2 * N_NEIGHBOURS))  # tell LightGBM these are words (categories), not amounts
BYTES_FROM_NEIGHBOURS = {"A alone": 0, "B network, raw": N_NEIGHBOURS * HISTORY, "C network, words": N_NEIGHBOURS * 2}

train_ends = ends_between(0, val_start, TRAIN_EVERY)
test_ends = ends_between(test_start, n_steps, TEST_EVERY)

# The picture's day: the test weekday + sensor with the deepest rush-hour drop
# (0 readings are missing data, not jams, so they're ignored).
known = np.where(speeds > 0, speeds, np.nan)
dates = times.normalize()
test_days = sorted({d for d in dates[test_start:] if d.dayofweek < 5})[1:-1]


def drop(rows):
    return np.nanmedian(known[rows], axis=0) - np.nanmin(known[rows], axis=0)


day = max(test_days, key=lambda d: drop(dates == d).max())
in_day = np.where(dates == day)[0]
sensor = int(np.argmax(drop(in_day)))
plot_ends = in_day - 12  # each forecast is made 60 minutes before the moment it predicts

# Step 4: train and test each input at each horizon.
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
print(f"{'input':<18}{'bytes from neighbours':>22}" + "".join(f"{h:>15}" for h in HORIZONS))
for name in INPUTS:
    print(f"{name:<18}{BYTES_FROM_NEIGHBOURS[name]:>22}"
          + "".join(f"{results[name, h][0]:>9.2f} ({results[name, h][1]:>3.0f}s)" for h in HORIZONS))
print(f"{'DCRNN (published)':<18}{'':>22}" + "".join(f"{v:>15.2f}" for v in (1.38, 1.74, 2.07)))

# Step 5: verdict against the bar fixed before running (60-minute forecast).
a, b, c = (results[n, "60 min"][0] for n in INPUTS)
helps = max(a - b, a - c) / a
keeps = (c - b) / b
ratio = BYTES_FROM_NEIGHBOURS["B network, raw"] / BYTES_FROM_NEIGHBOURS["C network, words"]
print("\nVerdict (60-minute forecast):")
print(f"  Network helps: best network input cuts A's error by {helps:.1%} (bar: >= 3%)  {'PASS' if helps >= 0.03 else 'FAIL'}")
print(f"  Words keep it: C vs B {keeps:+.1%} (bar: within +2%), neighbours send {ratio:.0f}x fewer bytes (bar: >= 5x)  "
      f"{'PASS' if keeps <= 0.02 and ratio >= 5 else 'FAIL'}")

# Step 6: picture. Left: one rush-hour day, actual vs 60-min forecasts. Right: error by horizon.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {"A alone": "#2a78d6", "B network, raw": "#eb6834", "C network, words": "#1baf7a"}
target_times = times[plot_ends + 12]
hours = target_times.hour + target_times.minute / 60
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.8, 1]}, facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

left.plot(hours, speeds[plot_ends + 12, sensor], color=INK, linewidth=2, label="actual speed")
for name, colour in COLOURS.items():
    left.plot(hours, curves[name], color=colour, linewidth=2, label=f"{name} (forecast 60 min earlier)")
left.set_xlim(hours[0], hours[-1])
left.set_xticks(range(0, 25, 3))
left.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
left.set_ylabel("speed (mph)", color=MUTED, fontsize=9)
left.set_title(f"Sensor {sensors[sensor]}, {day:%a %d %b %Y}: actual speed vs forecasts made 60 min earlier",
               loc="left", fontsize=10, color=INK)
left.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)

names = list(INPUTS) + ["DCRNN (published)"]
published = {"15 min": 1.38, "30 min": 1.74, "60 min": 2.07}
width = 0.2
for k, name in enumerate(names):
    values = [published[h] if name.startswith("DCRNN") else results[name, h][0] for h in HORIZONS]
    right.bar(np.arange(3) + (k - 1.5) * width, values, width=width - 0.03, label=name,
              color=COLOURS.get(name, "#a8a7a2"))
right.set_xticks(range(3))
right.set_xticklabels(list(HORIZONS))
right.set_ylabel("average error (mph), lower is better", color=MUTED, fontsize=9)
right.set_title("Error on the test months by forecast horizon", loc="left", fontsize=10, color=INK)
right.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
fig.tight_layout()
fig.savefig(PICTURE, dpi=130, facecolor=SURFACE)
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
