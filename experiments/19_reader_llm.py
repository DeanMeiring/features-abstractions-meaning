"""Experiment 19: reader LLM, phase 1. Can an off-the-shelf LLM read the language through its word cards?

Claude Opus 5.5 (Claude API) gets a test image's 8 words plus their word cards
from Library v0, and answers with the class, a confidence from 0 to 1, the words
that decided it, and a one-line reason (fam/reader.py). No training. Compared
with experiment 18's card vote (each word votes with its card's class shares)
on the same 500 test images. The developer's own tailored reader (phase 2) has
to beat this.

Success bar, fixed in the README before running:
    reads the language:  LLM accuracy no more than 3 points below the card vote on the same
                         images, whole 95% interval of the paired difference above -3 points
    grounded:            >= 95% of answers cite only words in that image's message
    knows when unsure:   accuracy at confidence >= 0.8 is >= 10 points above accuracy below 0.8
    reported, no bar:    cost, tokens, time

Needs experiment 18 first (data/library_v0.db). Answers are saved as they arrive
(data/reader/exp19_answers.jsonl), so a stopped run resumes without paying twice.
    python experiments/19_reader_llm.py --dry-run   # show one prompt, call nothing
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np

from fam import library, reader, scorecard
from fam.images import CLASSES

LIBRARY = ROOT / "data" / "library_v0.db"
ANSWERS = ROOT / "data" / "reader" / "exp19_answers.jsonl"
N_IMAGES = 500
PARALLEL = 4                        # requests in flight at once (waiting on the network, not the CPU)
PRICE_IN, PRICE_OUT = 4.00, 20.00   # $ per million tokens, Claude Opus 5.5

lib = library.Library(LIBRARY)
d = lib.dictionary("fashion-set8", "v1")["id"]
cards = {c["symbol"]: c for c in lib.cards(d)}
item_ids, words, labels = lib.messages(d, "test")
pick = np.sort(np.random.default_rng(19).choice(len(item_ids), size=N_IMAGES, replace=False))

if "--dry-run" in sys.argv:
    print("SYSTEM:\n" + reader.system_prompt(CLASSES) + "\n")
    print("USER (first sampled image, true class " + CLASSES[labels[pick[0]]] + "):\n" + reader.item_text(words[pick[0]], cards))
    print(f"\n~{(len(reader.system_prompt(CLASSES)) + len(reader.item_text(words[pick[0]], cards))) // 4} input tokens per image (rough)")
    sys.exit()

# Step 1: ask Claude about each sampled image, skipping ones already answered.
ANSWERS.parent.mkdir(parents=True, exist_ok=True)
done = {}
if ANSWERS.exists():
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        done[row["item_id"]] = row
todo = [i for i in pick if int(item_ids[i]) not in done]
print(f"{len(done)} answers already saved, {len(todo)} to ask")

api = reader.client()
start = time.time()


def ask(i):
    answer = reader.read(api, words[i], cards, CLASSES)
    return {"item_id": int(item_ids[i]), "true_class": CLASSES[labels[i]], **answer}


with ThreadPoolExecutor(PARALLEL) as pool, ANSWERS.open("a", encoding="utf-8") as out:
    for n, row in enumerate(pool.map(ask, todo), 1):
        out.write(json.dumps(row) + "\n")
        out.flush()
        done[row["item_id"]] = row
        if n % 50 == 0:
            print(f"  {n}/{len(todo)} answered ({time.time() - start:.0f} s)")

# Step 2: score against the true labels, and against the card vote on the same images.
rows = [done[int(item_ids[i])] for i in pick]
answered = [r for r in rows if not r["refused"]]
llm_right = np.array([(not r["refused"]) and r["class"] == r["true_class"] for r in rows], dtype=float)

shares = np.zeros((256, len(CLASSES)))
for s, card in cards.items():
    shares[s] = [card["class_counts"][c] / card["uses"] for c in CLASSES]
vote_right = (shares[words[pick]].sum(axis=1).argmax(axis=1) == labels[pick]).astype(float)

llm, llm_lo, llm_hi = scorecard.interval(llm_right)
vote, vote_lo, vote_hi = scorecard.interval(vote_right)
gap, gap_lo, gap_hi = scorecard.interval(llm_right - vote_right)
grounded = np.mean([set(r["key_words"]) <= set(int(w) for w in words[np.where(item_ids == r["item_id"])[0][0]])
                    for r in answered])
sure = np.array([r["confidence"] >= 0.8 for r in answered])
right = np.array([r["class"] == r["true_class"] for r in answered])
sure_acc = right[sure].mean() if sure.any() else float("nan")
unsure_acc = right[~sure].mean() if (~sure).any() else float("nan")

tokens_in = sum(r["input_tokens"] for r in rows)
tokens_out = sum(r["output_tokens"] for r in rows)
cost = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT

print(f"\nOn {N_IMAGES} test images ({sum(r['refused'] for r in rows)} refused):")
print(f"  LLM reading word cards:  {llm:.1%} [{llm_lo:.1%}, {llm_hi:.1%}]")
print(f"  card vote (experiment 18): {vote:.1%} [{vote_lo:.1%}, {vote_hi:.1%}]")
print(f"  confidence >= 0.8 on {sure.sum()} answers: {sure_acc:.1%} right; below 0.8 on {(~sure).sum()}: {unsure_acc:.1%}")
print(f"  tokens: {tokens_in:,} in, {tokens_out:,} out, about ${cost:.2f} (Opus 5.5 list prices)")

print("\nThree example answers:")
for r in answered[:3]:
    print(f"  true {r['true_class']}: said {r['class']} ({r['confidence']:.2f}), "
          f"key words {' '.join(f'<w{w}>' for w in r['key_words'])}: {r['reason']}")

print("\nVerdict:")
print(f"  Reads the language: LLM minus vote {gap:+.1%} [{gap_lo:+.1%}, {gap_hi:+.1%}] (bar: whole interval above -3 points)  "
      f"{scorecard.verdict(gap_lo, gap_hi, -0.03)}")
print(f"  Grounded: {grounded:.1%} of answers cite only the image's own words (bar: >= 95%)  "
      f"{'PASS' if grounded >= 0.95 else 'FAIL'}")
calibrated = sure_acc - unsure_acc
print(f"  Knows when unsure: sure minus unsure accuracy {calibrated:+.1%} (bar: >= +10 points)  "
      f"{'PASS' if calibrated >= 0.10 else 'FAIL'}")
