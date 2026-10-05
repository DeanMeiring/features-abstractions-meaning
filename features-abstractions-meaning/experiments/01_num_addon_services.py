"""Experiment 1: does `num_addon_services` beat the raw-data baseline?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fam.data import evaluate, load_telco

ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def num_addon_services(X):
    """How many of the 6 internet add-ons the customer has (0-6)."""
    return (X[ADDONS] == "Yes").sum(axis=1)


X, y = load_telco()

baseline = evaluate(X, y)

X_new = X.copy()
X_new["num_addon_services"] = num_addon_services(X)
with_feature = evaluate(X_new, y)

diff = with_feature - baseline  # same splits, so we can compare split by split
print(f"Baseline (19 raw columns):      AUC {baseline.mean():.4f}  (± {baseline.std():.4f})")
print(f"+ num_addon_services:           AUC {with_feature.mean():.4f}  (± {with_feature.std():.4f})")
print(f"Difference:                     {diff.mean():+.4f}  (better on {(diff > 0).sum()}/15 splits)")

# How churn changes with the feature value: does it carry a real signal?
print("\nChurn rate by num_addon_services:")
rates = y.groupby(X_new["num_addon_services"]).agg(["mean", "size"])
for n, row in rates.iterrows():
    print(f"  {n}: {row['mean']:.0%} churn  ({int(row['size'])} customers)")
