"""Load PEMS-BAY and set up the standard forecasting task the same way every time.

PEMS-BAY: 325 road sensors (our "points"), average speed in mph every 5 minutes,
Jan-Jun 2017. The standard task (DCRNN, Li et al. 2018): from the last hour
(12 readings) predict a sensor's speed 15, 30 and 60 minutes ahead
(3, 6, 12 steps). Split by time: first 70% train, next 10% validation,
last 20% test, so the test is always the future.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "pems_bay"
HISTORY = 12                       # the last hour, as input
HORIZONS = {"15 min": 3, "30 min": 6, "60 min": 12}


def load_speeds() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Return speeds (time, sensors) as float32 mph, the timestamps and the sensor ids."""
    df = pd.read_csv(RAW / "pems_bay.csv", index_col=0, parse_dates=True)
    return df.values.astype(np.float32), df.index, df.columns.astype(int).values


def load_road_links(sensor_ids: np.ndarray) -> pd.DataFrame:
    """Road distances (metres) between pairs of sensors: the connections between points."""
    links = pd.read_csv(RAW / "distances_bay_2017.csv", header=None, names=["from", "to", "metres"])
    known = links["from"].isin(sensor_ids) & links["to"].isin(sensor_ids)
    return links[known & (links["from"] != links["to"])].reset_index(drop=True)


def road_distances(sensor_ids: np.ndarray) -> np.ndarray:
    """drive[i, j]: metres from sensor i to sensor j along the road (inf if not linked)."""
    index = {s: i for i, s in enumerate(sensor_ids)}
    drive = np.full((len(sensor_ids), len(sensor_ids)), np.inf)
    for a, b, metres in load_road_links(sensor_ids)[["from", "to", "metres"]].itertuples(index=False):
        drive[index[a], index[b]] = min(drive[index[a], index[b]], metres)
    return drive


def road_weights(sensor_ids: np.ndarray, keep: float = 0.1) -> np.ndarray:
    """How strongly each pair of sensors is linked, from road distance (as in DCRNN):
    w = exp(-(distance / spread)^2), so close sensors ~1 and far ones ~0; weak links
    (below `keep`) are dropped. Directional: w[i, j] follows the road from i to j."""
    drive = road_distances(sensor_ids)
    spread = drive[np.isfinite(drive)].std()
    weights = np.where(np.isfinite(drive), np.exp(-(drive / spread) ** 2), 0.0)
    weights[weights < keep] = 0.0
    np.fill_diagonal(weights, 0.0)
    return weights


def day_bootstrap_gain(err_without: np.ndarray, err_with: np.ndarray, days: np.ndarray,
                       draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Relative error reduction from adding neighbours, 1 - error_with / error_without,
    with a 95% interval from resampling whole days (readings within a day are linked,
    so resampling single readings would make the interval look falsely tight)."""
    unique, day_index = np.unique(days, return_inverse=True)
    without = np.bincount(day_index, weights=err_without, minlength=len(unique))
    with_ = np.bincount(day_index, weights=err_with, minlength=len(unique))
    picks = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique)))
    gains = 1 - with_[picks].sum(axis=1) / without[picks].sum(axis=1)
    return float(1 - with_.sum() / without.sum()), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def nearest_neighbours(sensor_ids: np.ndarray, k: int = 5, direction: str = "ahead") -> tuple[np.ndarray, int]:
    """Each sensor's k nearest sensors along the road, in one direction of traffic.

    ahead:  sensors you drive TO (downstream), where a jam's queue starts
    behind: sensors that drive TO you (upstream), where your traffic comes from
    any:    either way (experiment 12)
    The road distances are directional (from -> to). When two sensors are linked
    both ways, the shorter route decides which way is "ahead". Sensors with fewer
    than k linked in that direction get their gaps filled by map distance
    (ranked after every road-linked sensor). Returns (neighbours (sensors, k),
    how many sensors needed filling).
    """
    drive = road_distances(sensor_ids)
    ahead = np.where(drive <= drive.T, drive, np.inf)  # keep each link in its shorter direction only
    road = {"ahead": ahead, "behind": ahead.T, "any": np.minimum(drive, drive.T)}[direction]

    meta = pd.read_csv(RAW / "pems_bay_meta.csv").set_index("sensor_id").loc[sensor_ids]
    lat, lon = np.radians(meta["Latitude"].values), np.radians(meta["Longitude"].values)
    on_map = 6_371_000 * np.hypot(lat[:, None] - lat[None, :], (lon[:, None] - lon[None, :]) * np.cos(lat.mean()))
    ranking = np.where(np.isfinite(road), road, 1e9 + on_map)
    np.fill_diagonal(ranking, np.inf)
    filled = int((np.isfinite(road).sum(axis=1) < k).sum())
    return np.argsort(ranking, axis=1)[:, :k], filled


def split_points(n_steps: int) -> tuple[int, int]:
    """Time steps where validation and test start (70% / 10% / 20%)."""
    return round(n_steps * 0.7), round(n_steps * 0.8)


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error in mph, ignoring the few readings that are 0 (missing), as the benchmark does."""
    known = actual > 0
    return float(np.abs(predicted[known] - actual[known]).mean())
