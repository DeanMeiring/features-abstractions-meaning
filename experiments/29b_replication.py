"""Experiment 29b: does experiment 29 replicate? The verdict from its two runs.

Run the two replications first (each ~40 min; experiment 29's script with an option):
    python experiments/29_profile_words.py --seed 1        # run S: another training seed
    python experiments/29_profile_words.py --later-week    # run W: train 11-24 Nov, test 25 Nov - 1 Dec
then:
    python experiments/29b_replication.py                  # reads their saved errors, applies the bar

Bar (README, committed before this was built):
    replicated:     in both runs, (1) R+P / R - 1 whole interval below 0 on >= 3 of 4 questions,
                    and (2) on both next-day questions, the whole interval below -3%
    not replicated: in either run, R+P / R - 1 whole interval above 0 on >= 2 of 4,
                    or a next-day whole interval entirely above -3%
    else:           too close to call (nothing builds on it); no reruns
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fam import telecom  # noqa: E402

RUNS = {"S (seed 1)": "exp29_predictions_seed1.npz", "W (later week)": "exp29_predictions_laterweek.npz"}
QUESTIONS = ["next_hour_internet", "next_hour_calls", "next_day_internet", "next_day_calls"]

replicated, failed = [], []
for run, name in RUNS.items():
    saved = np.load(ROOT / "data" / name)
    blocks = saved["blocks"]
    print(f"\nRun {run}:")
    gains, worse, next_day_ok, next_day_small = 0, 0, 0, 0
    for q in QUESTIONS:
        point, lo, hi = telecom.block_interval(saved[f"err_R+P_{q}"], saved[f"err_R_{q}"], blocks)
        vs_h = telecom.block_interval(saved[f"err_R+P_{q}"], saved[f"err_R+H_{q}"], blocks)
        all_vs_h = telecom.block_interval(saved[f"err_R+H+P_{q}"], saved[f"err_R+H_{q}"], blocks)
        gains += hi < 0
        worse += lo > 0
        if q.startswith("next_day"):
            next_day_ok += hi < -0.03
            next_day_small += lo > -0.03
        print(f"  {q:<20} R+P vs R {point:+.1%} [{lo:+.1%}, {hi:+.1%}]   "
              f"(reported) R+P vs R+H {vs_h[0]:+.1%} [{vs_h[1]:+.1%}, {vs_h[2]:+.1%}]   "
              f"R+H+P vs R+H {all_vs_h[0]:+.1%}")
    print(f"  (1) words beat raw on {gains} of 4 (needs 3); (2) next-day intervals below -3%: {next_day_ok} of 2 (needs 2)")
    replicated.append(gains >= 3 and next_day_ok == 2)
    failed.append(worse >= 2 or next_day_small >= 1)

if all(replicated):
    verdict = "REPLICATED: experiment 29's gain holds on another seed and another week"
elif any(failed):
    verdict = "NOT REPLICATED: experiment 29's gain doesn't hold"
else:
    verdict = "TOO CLOSE TO CALL: not shown; nothing builds on experiment 29 yet"
print(f"\nVerdict: {verdict}")
