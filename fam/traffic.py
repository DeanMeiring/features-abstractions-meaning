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


def split_points(n_steps: int) -> tuple[int, int]:
    """Time steps where validation and test start (70% / 10% / 20%)."""
    return round(n_steps * 0.7), round(n_steps * 0.8)


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error in mph, ignoring the few readings that are 0 (missing), as the benchmark does."""
    known = actual > 0
    return float(np.abs(predicted[known] - actual[known]).mean())
