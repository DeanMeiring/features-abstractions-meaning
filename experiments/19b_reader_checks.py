"""Experiment 19b: does the reader LLM add anything beyond the card vote?

Two free checks on experiment 19's saved answers (data/reader/exp19_answers.jsonl):
no new API calls, no training, no new encoder. Same 500 test images.

    Check A  does the LLM just copy the vote?  Share of images where the LLM's class
             equals the card vote's class. If the whole 95% interval is >= 95%, the LLM
             reads the cards the way the vote does; otherwise, where they disagree, who
             is right more often?
    Check B  is the LLM's confidence just the vote's margin?  Vote margin = top class
             share minus runner-up share. The 280 images with the highest margin vs the
             LLM's 280 answers with confidence >= 0.8: which set is more often right?
             Paired per image over all 500: (in LLM set and right) - (in margin set and
             right), times 500/280, whose mean is the accuracy difference of the two sets.
             The LLM adds something only if the whole 95% interval is above 0.

Bars fixed in the README before computing. Needs experiments 18 and 19 first.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np

from fam import library, scorecard
from fam.images import CLASSES

lib = library.Library(ROOT / "data" / "library_v0.db")
d = lib.dictionary("fashion-set8", "v1")["id"]
cards = {c["symbol"]: c for c in lib.cards(d)}
item_ids, words, labels = lib.messages(d, "test")
rows = [json.loads(line) for line in (ROOT / "data" / "reader" / "exp19_answers.jsonl").open(encoding="utf-8")]
rows.sort(key=lambda r: r["item_id"])  # image order, for tie-breaking
index = {int(i): n for n, i in enumerate(item_ids)}
pos = np.array([index[r["item_id"]] for r in rows])

# The card vote on the same images, and its margin: top class share minus runner-up share.
shares = np.zeros((256, len(CLASSES)))
for s, card in cards.items():
    shares[s] = [card["class_counts"][c] / card["uses"] for c in CLASSES]
votes = shares[words[pos]].mean(axis=1)                 # (500, classes): each word's shares, averaged
vote_class = votes.argmax(axis=1)
top_two = np.sort(votes, axis=1)[:, -2:]
margin = top_two[:, 1] - top_two[:, 0]

truth = labels[pos]
llm_class = np.array([CLASSES.index(r["class"]) for r in rows])
confidence = np.array([r["confidence"] for r in rows])
llm_right, vote_right = llm_class == truth, vote_class == truth

# Check A: agreement, and who wins where they disagree.
agree, agree_lo, agree_hi = scorecard.interval((llm_class == vote_class).astype(float))
print(f"Check A: the LLM picks the vote's class on {agree:.1%} [{agree_lo:.1%}, {agree_hi:.1%}] of {len(rows)} images")
if agree_lo >= 0.95:
    print("  -> the LLM reads the cards the same way the vote does; it adds explanations, not judgement.")
else:
    split = llm_class != vote_class
    l, l_lo, l_hi = scorecard.interval(llm_right[split].astype(float))
    v, v_lo, v_hi = scorecard.interval(vote_right[split].astype(float))
    g, g_lo, g_hi = scorecard.interval(llm_right[split].astype(float) - vote_right[split])
    print(f"  Where they disagree ({split.sum()} images): LLM right {l:.1%} [{l_lo:.1%}, {l_hi:.1%}], "
          f"vote right {v:.1%} [{v_lo:.1%}, {v_hi:.1%}], LLM minus vote {g:+.1%} [{g_lo:+.1%}, {g_hi:+.1%}]")

# Check B: the LLM's 280 confident answers vs the vote's 280 highest margins.
n = int((confidence >= 0.8).sum())
llm_set = confidence >= 0.8
margin_set = np.zeros(len(rows), dtype=bool)
margin_set[np.argsort(-margin, kind="stable")[:n]] = True
paired = ((llm_set & llm_right).astype(float) - (margin_set & vote_right)) * len(rows) / n
diff, diff_lo, diff_hi = scorecard.interval(paired)
overlap = (llm_set & margin_set).sum()

print(f"\nCheck B: the {n} most confident answers ({overlap} of them are the same images in both sets)")
print(f"  {'':<28}{'top ' + str(n):>12}{'other ' + str(len(rows) - n):>12}")
print(f"  {'LLM confidence (>= 0.8)':<28}{llm_right[llm_set].mean():>12.1%}{llm_right[~llm_set].mean():>12.1%}")
print(f"  {'vote margin (highest)':<28}{vote_right[margin_set].mean():>12.1%}{vote_right[~margin_set].mean():>12.1%}")
print(f"  LLM minus vote margin, top {n}: {diff:+.1%} [{diff_lo:+.1%}, {diff_hi:+.1%}]  "
      + ("-> the LLM's confidence adds something" if diff_lo > 0 else "-> the confidence comes from the cards, not the LLM"))
