"""Experiment 3: can customers be written as 4 words from a learned vocabulary?

Same setup as experiment 2 (model B predicts long contract, Contract column
hidden), but now model A is a VQ-VAE: each customer becomes 4 word numbers
(0-255), i.e. 4 bytes. Model B reads those words by looking them up in the
shared codebook.

We compare model B with:
  raw        the raw columns (the baseline)
  ae codes   experiment 2's 4 continuous numbers
  vq words   4 words, looked up in the codebook

Then we measure storage size against the raw data, including zipped.
"""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 45%)

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from fam.autoencoder import CustomerEncoder
from fam.data import MODEL_PARAMS, SEED, load_telco
from fam.vqvae import CustomerWordEncoder

LABEL_SIZES = [50, 100, 300, 1000, None]  # None = all training labels

X, _ = load_telco()
y = (X["Contract"] != "Month-to-month").astype(int)
X = X.drop(columns=["Contract"])

splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
results = []

for split_no, (train_idx, test_idx) in enumerate(splits.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    ae = CustomerEncoder(code_size=4).fit(X_train)
    vq = CustomerWordEncoder().fit(X_train)

    inputs = {
        "raw": (X_train, X_test),
        "ae codes": (ae.encode(X_train), ae.encode(X_test)),
        "vq words": (vq.encode(X_train), vq.encode(X_test)),
    }

    for n in LABEL_SIZES:
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

    print(
        f"split {split_no + 1}/10 done "
        f"(rebuild error: ae {ae.final_loss:.3f}, vq {vq.final_loss:.3f}; "
        f"vq uses {vq.words_used}/256 words)"
    )

table = (
    pd.DataFrame(results)
    .groupby(["labels", "input"])["auc"]
    .mean()
    .unstack()
    .loc[[str(n or "all") for n in LABEL_SIZES], ["raw", "ae codes", "vq words"]]
)
print("\nModel B test AUC (long contract?), averaged over 10 splits:\n")
print(table.round(3).to_string())

# Storage: what you'd keep on disk to describe all 7,043 customers.
vq = CustomerWordEncoder().fit(X)
words = vq.words(X)
raw_zip = len(gzip.compress(X.to_csv(index=False).encode()))
words_bytes = words.to_numpy().nbytes  # 4 bytes per customer
words_zip = len(gzip.compress(words.to_numpy().tobytes()))
codebook_bytes = vq.model.codebook.detach().numpy().nbytes  # the shared dictionary, stored once

print("\nStorage for all customers:")
print(f"  raw data, zipped:        {raw_zip / 1024:6.1f} KB")
print(f"  vq words (4 bytes each): {words_bytes / 1024:6.1f} KB  ({raw_zip / words_bytes:.1f}x smaller)")
print(f"  vq words, zipped:        {words_zip / 1024:6.1f} KB  ({raw_zip / words_zip:.1f}x smaller)")
print(f"  + codebook (shared, once): {codebook_bytes / 1024:4.1f} KB")
print(f"\nExample customers as words:\n{words.head(5).to_string()}")
