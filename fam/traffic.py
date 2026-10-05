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


def nearest_neighbours(sensor_ids: np.ndarray, k: int = 5, direction: str = "ahead") -> tuple[np.ndarray, int]:
    """Each sensor's k nearest sensors along the road, in one direction of traffic.

    ahead:  sensors you drive TO (downstream), where a jam's queue starts
    behind: sensors that drive TO you (upstream), where your traffic comes from
    The road distances are directional (from -> to). When two sensors are linked
    both ways, the shorter route decides which way is "ahead". Sensors with fewer
    than k linked in that direction get their gaps filled by map distance
    (ranked after every road-linked sensor). Returns (neighbours (sensors, k),
    how many sensors needed filling).
    """
    index = {s: i for i, s in enumerate(sensor_ids)}
    n = len(sensor_ids)
    drive = np.full((n, n), np.inf)  # drive[i, j]: metres from sensor i to sensor j along the road
    for a, b, metres in load_road_links(sensor_ids)[["from", "to", "metres"]].itertuples(index=False):
        drive[index[a], index[b]] = min(drive[index[a], index[b]], metres)
    ahead = np.where(drive <= drive.T, drive, np.inf)  # keep each link in its shorter direction only
    road = ahead if direction == "ahead" else ahead.T

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
