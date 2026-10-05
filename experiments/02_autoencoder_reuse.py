"""Experiment 2: do codes learned by an autoencoder (model A) help a model on a new task (model B)?

Model A: an autoencoder learns a 4-number code per customer from the raw
         columns. It never sees any label, only customer data.
Model B: LightGBM predicts a different task: is the customer on a long
         contract (one or two year) instead of month-to-month?

We compare model B with three inputs:
  raw        the raw columns (the baseline)
  raw+codes  the raw columns plus model A's 4 codes
  codes      only model A's 4 codes (can 4 numbers stand in for 18 columns?)

And with different amounts of labelled data, because the main promise of
reusing learned features is needing fewer labels. Model A trains on all
training customers (unlabelled data is cheap); model B gets only N labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from fam.autoencoder import CustomerEncoder
from fam.data import MODEL_PARAMS, SEED, load_telco

LABEL_SIZES = [50, 100, 300, 1000, None]  # None = all training labels

X, _ = load_telco()

# Task B label. Contract is removed from the inputs, otherwise the answer is
# sitting in the data (leakage). Model A doesn't see it either, for the same reason.
y = (X["Contract"] != "Month-to-month").astype(int)
X = X.drop(columns=["Contract"])

splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
results = []

for split_no, (train_idx, test_idx) in enumerate(splits.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Model A learns from training customers only; test customers stay unseen.
    encoder = CustomerEncoder(code_size=4).fit(X_train)
    codes_train, codes_test = encoder.encode(X_train), encoder.encode(X_test)

    inputs = {
        "raw": (X_train, X_test),
        "raw+codes": (X_train.join(codes_train), X_test.join(codes_test)),
        "codes": (codes_train, codes_test),
    }

    for n in LABEL_SIZES:
        # Pick n labelled customers (same ones for every input type, for fairness).
        if n is None:
            pick = X_train.index
        else:
            pick, _ = train_test_split(
                X_train.index, train_size=n, stratify=y_train, random_state=split_no
            )

        for name, (inp_train, inp_test) in inputs.items():
            model = lgb.LGBMClassifier(**MODEL_PARAMS)
            model.fit(inp_train.loc[pick], y_train.loc[pick])
            auc = roc_auc_score(y_test, model.predict_proba(inp_test)[:, 1])
            results.append({"labels": str(n or "all"), "input": name, "auc": auc})

    print(f"split {split_no + 1}/10 done (autoencoder rebuild error {encoder.final_loss:.3f})")

table = (
    pd.DataFrame(results)
    .groupby(["labels", "input"])["auc"]
    .mean()
    .unstack()
    .loc[[str(n or "all") for n in LABEL_SIZES], ["raw", "raw+codes", "codes"]]
)
print("\nModel B test AUC (long contract?), averaged over 10 splits:\n")
print(table.round(3).to_string())
