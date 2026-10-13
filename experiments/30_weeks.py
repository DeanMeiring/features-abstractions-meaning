"""Experiment 30: does the profile gain hold across ordinary weeks? The verdict from the weekly runs.

Run each test week first (experiment 29's script, ~40 min each; the first also builds the 56-day cache):
    python experiments/29_profile_words.py --weeks-later 2    # test week Mon 2 Dec   (counted)
    python experiments/29_profile_words.py --weeks-later 3    # test week Mon 9 Dec   (counted)
    python experiments/29_profile_words.py --weeks-later 4    # test week Mon 16 Dec  (counted)
    python experiments/29_profile_words.py --weeks-later 5    # test week Mon 23 Dec  (Christmas, reported only)
then:
    python experiments/30_weeks.py                            # reads their saved errors, applies the bar

Bar (README, committed before this was built):
    present in a week: on both next-day questions, R+P / R - 1 whole interval below -3%
                       (a week whose dictionary failed its health gate has no saved errors: not present)
    established: present in all 3 counted weeks · 2 of 3: too close to call · 1 or 0: not established
    reported only: 18 Nov (experiment 29), 25 Nov (experiment 29b run W), 23 Dec (Christmas)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fam import telecom  # noqa: E402

WEEKS = [  # (test week, saved errors, counted?)
    ("25 Nov", "exp29_predictions_laterweek.npz", False),
    ("2 Dec", "exp29_predictions_weeks2.npz", True),
    ("9 Dec", "exp29_predictions_weeks3.npz", True),
    ("16 Dec", "exp29_predictions_weeks4.npz", True),
    ("23 Dec", "exp29_predictions_weeks5.npz", False),
]
QUESTIONS = ["next_hour_internet", "next_hour_calls", "next_day_internet", "next_day_calls"]

print("18 Nov (experiment 29, reported only): next day -6.1% [-7.3, -5.0] / -6.4% [-7.2, -5.6] (see the README)")
present = 0
for week, name, counted in WEEKS:
    path = ROOT / "data" / name
    label = f"{week} ({'counted' if counted else 'reported only'})"
    if not path.exists():
        print(f"\n{label}: no saved errors (health gate failed, or not run): not present")
        continue
    saved = np.load(path)
    blocks = saved["blocks"]
    print(f"\n{label}:")
    below = 0
    for q in QUESTIONS:
        point, lo, hi = telecom.block_interval(saved[f"err_R+P_{q}"], saved[f"err_R_{q}"], blocks)
        vs_h = telecom.block_interval(saved[f"err_R+P_{q}"], saved[f"err_R+H_{q}"], blocks)
        h_vs_r = telecom.block_interval(saved[f"err_R+H_{q}"], saved[f"err_R_{q}"], blocks)
        if q.startswith("next_day"):
            below += hi < -0.03
        print(f"  {q:<20} R+P vs R {point:+.1%} [{lo:+.1%}, {hi:+.1%}]   "
              f"(reported) R+P vs R+H {vs_h[0]:+.1%}   R+H vs R {h_vs_r[0]:+.1%}")
    is_present = below == 2
    print(f"  present (both next-day intervals below -3%): {'yes' if is_present else 'no'}")
    present += counted and is_present

print(f"\nPresent in {present} of 3 counted weeks (2, 9, 16 Dec)")
verdict = ("ESTABLISHED: profile context helps across ordinary weeks" if present == 3 else
           "TOO CLOSE TO CALL: not established for building on it" if present == 2 else
           "NOT ESTABLISHED")
print(f"Verdict: {verdict}")
