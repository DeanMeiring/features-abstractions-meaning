"""Load the Telecom Italia Milan data as hourly activity per grid square.

Raw files (scripts/download_telecom.py): one per day, tab-separated, one row per
(grid square, 10-minute interval, country code) with SMS in/out, calls in/out and
internet activity. Blank readings mean no activity.

Here: summed over the six 10-minute readings and all country codes into hourly
totals of SMS (in + out), calls (in + out) and internet, for each of the 10,000
squares (a 100 x 100 grid, numbered row by row from 1). Missing rows and blanks
count as 0. Hours are Milan local time from Mon 2013-11-04 00:00 (no clock change
in November). Reading 7.3 GB of text takes several minutes, so the result is
saved to data/telecom_hourly.npz and reused.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "telecom_milan"
CACHE = ROOT / "data" / "telecom_hourly.npz"
START_MS = 1383519600000           # 2013-11-04 00:00 Milan time (23:00 UTC the day before)
DAYS, SQUARES = 21, 10000
HOURS = DAYS * 24
TRAIN_HOURS = 14 * 24              # Mon 4 - Sun 17 Nov; the test week is Mon 18 - Sun 24 Nov
ACTIVITIES = ["sms", "calls", "internet"]
COLUMNS = ["square", "time", "country", "sms_in", "sms_out", "call_in", "call_out", "internet"]


def load_hourly() -> np.ndarray:
    """Activity counts (squares, hours, 3 activities) as float32, before any log."""
    if CACHE.exists():
        return np.load(CACHE)["counts"]
    counts = np.zeros((SQUARES, HOURS, len(ACTIVITIES)), dtype=np.float64)
    for path in sorted(RAW.glob("sms-call-internet-mi-2013-11-*.txt")):
        day = pd.read_csv(path, sep="\t", header=None, names=COLUMNS).fillna(0.0)
        hour = ((day["time"] - START_MS) // 3_600_000).to_numpy()
        square = day["square"].to_numpy() - 1
        keep = (hour >= 0) & (hour < HOURS)
        values = np.column_stack([day["sms_in"] + day["sms_out"], day["call_in"] + day["call_out"], day["internet"]])
        np.add.at(counts, (square[keep], hour[keep]), values[keep])
        print(f"  read {path.name}")
    counts = counts.astype(np.float32)
    np.savez_compressed(CACHE, counts=counts)
    return counts


def spatial_blocks() -> np.ndarray:
    """Block number (0-99) of each square: the 100 blocks of 10 x 10 neighbouring squares."""
    square = np.arange(SQUARES)
    row, col = square // 100, square % 100
    return (row // 10) * 10 + (col // 10)


def block_interval(err_a: np.ndarray, err_b: np.ndarray, blocks: np.ndarray, draws: int = 2000,
                   seed: int = 0) -> tuple[float, float, float]:
    """mean(err_a) / mean(err_b) - 1, with a 95% interval from resampling whole spatial blocks.

    err_a, err_b: one error per test example (same examples); blocks: each example's block.
    Neighbouring squares behave alike, so whole blocks are resampled, not single squares.
    """
    n_blocks = blocks.max() + 1
    sum_a = np.bincount(blocks, weights=err_a, minlength=n_blocks)
    sum_b = np.bincount(blocks, weights=err_b, minlength=n_blocks)
    picks = np.random.default_rng(seed).integers(n_blocks, size=(draws, n_blocks))
    ratios = sum_a[picks].sum(axis=1) / sum_b[picks].sum(axis=1) - 1
    return float(sum_a.sum() / sum_b.sum() - 1), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))
