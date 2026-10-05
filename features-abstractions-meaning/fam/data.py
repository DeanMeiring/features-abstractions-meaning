"""Load the Telco data and evaluate feature sets the same way every time."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"
SEED = 42

# Model settings are fixed so that only the features change between experiments.
MODEL_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    random_state=SEED,
    verbose=-1,
)


def load_telco() -> tuple[pd.DataFrame, pd.Series]:
    """Return the 19 raw input columns (X) and the churn label (y, 1 = churned)."""
    df = pd.read_csv(DATA)

    # 11 brand-new customers (tenure 0) have a blank TotalCharges: they haven't paid yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    # LightGBM handles text categories itself if we mark them as 'category'.
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    return X, y


def evaluate(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Train and test LightGBM on 15 fixed splits; return the 15 test AUCs.

    One 80/20 split gives a noisy AUC (it depends on which customers land in
    the test set). Repeating 5-fold cross-validation 3 times averages that
    luck out. The splits are seeded, so every experiment sees the same ones.
    """
    splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    aucs = []
    for train_idx, test_idx in splits.split(X, y):
        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], proba))
    return np.array(aucs)
