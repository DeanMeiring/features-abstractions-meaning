"""Experiment 15: can a new model learn the existing language?

Experiment 14 trained the same recipe twice from scratch and got two half-
different languages. But real languages aren't reinvented by every speaker:
a new speaker learns the existing one by trying to be understood.

So: freeze experiment 14's dictionary (the symbols and the drawer that turns
symbols back into an image) and train a brand-new encoder, the "new speaker",
from a fresh random start. Its only job is to describe images so the frozen
drawer can rebuild them. It never sees what the original speaker wrote.
Done for the alphabet (the bars) and for flat words (comparison).

Model B reads each message with its symbols sorted by number (a message is an
unordered set, so slot order carries no meaning); the slot-order reader is
reported too.

Tests on the 10,000 test images:
    same symbols          share of symbols both speakers write for the same image
    understood by others  model B trained on the ORIGINAL speaker's messages
                          reads the NEW speaker's messages: accuracy lost?
    quality kept          the new speaker's own few-label accuracy

Success bar, fixed in the README before running (alphabet, 50 labels, 20 picks):
    same symbols >= 70%; understood by others: loses <= 2 points;
    quality kept: within 1 point of the original speaker.
Needs experiments 10 and 14 first.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch

from fam import scorecard, set_vqvae
from fam.images import load_fashion_mnist

MODELS = ROOT / "data" / "models"
DICTIONARIES = {  # recipe: (original speaker = the dictionary, new speaker's file)
    "alphabet": ("fashion_alpha8.pt", "fashion_alpha8_speaker7.pt"),
    "flat": ("fashion_set8.pt", "fashion_set8_speaker7.pt"),
}
start_all = time.time()

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
rng = np.random.default_rng(42)  # same picks as experiments 6-14 for the 50-label size
picks = {n: [rng.choice(len(train_images), size=n, replace=False) for _ in range(r)] for n, r in {50: 20, 1000: 5}.items()}


def vectors(model, words, sort=True):
    """Symbols -> their dictionary vectors. Sorted by symbol number by default: the
    message is an unordered set, so the same set must look the same to model B."""
    if sort:
        words = np.sort(words, axis=1)
    with torch.no_grad():
        return model.vectors()[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


results = {}
for recipe, (original_file, speaker_file) in DICTIONARIES.items():
    alphabet = recipe == "alphabet"
    original = set_vqvae.SetVQVAE(alphabet=alphabet)
    original.load_state_dict(torch.load(MODELS / original_file))
    original.eval()

    # Step 1: the new speaker learns the frozen dictionary (cached).
    if (MODELS / speaker_file).exists():
        speaker = set_vqvae.SetVQVAE(alphabet=alphabet)
        speaker.load_state_dict(torch.load(MODELS / speaker_file))
        print(f"Loaded {speaker_file}")
    else:
        print(f"Training a new speaker for the frozen {recipe} dictionary (fresh start, seed 7)...")
        t = time.time()
        speaker = set_vqvae.train(train_images, seed=7, dictionary=original)
        print(f"  took {time.time() - t:.0f} s")
        torch.save(speaker.state_dict(), MODELS / speaker_file)
    speaker.eval()
    assert torch.equal(speaker.vectors(), original.vectors()), "the dictionary must not change"

    # Step 2: both speakers describe the same images.
    orig_train, orig_test = set_vqvae.words_of(original, train_images), set_vqvae.words_of(original, test_images)
    new_train, new_test = set_vqvae.words_of(speaker, train_images), set_vqvae.words_of(speaker, test_images)
    with torch.no_grad():
        rebuilt = speaker.decode_words(torch.tensor(new_test, dtype=torch.long)).squeeze(1).numpy()

    row = {
        "same symbols": scorecard.same_symbols(orig_test, new_test),
        "same by chance": scorecard.same_symbols(orig_test, new_test[np.random.default_rng(0).permutation(len(new_test))]),
        "stability (exp 14 measure)": scorecard.stability(orig_test, new_test),
        "new speaker rebuild error": float(((rebuilt - test_images / 255.0) ** 2).mean()),
    }
    if alphabet:
        row["same radicals"] = scorecard.same_symbols(orig_test // 16, new_test // 16, vocab=16)

    # Step 3: model B, trained on the ORIGINAL speaker's messages, reads both speakers.
    o_train, o_test, n_train, n_test = (vectors(original, w) for w in (orig_train, orig_test, new_train, new_test))
    slot_order = [vectors(original, w, sort=False) for w in (orig_train, orig_test, new_test)]
    row["slot-order reader: original reads new, 50"] = scorecard.few_label_accuracy(
        slot_order[0], train_labels, slot_order[2], test_labels, picks[50])
    row["slot-order reader: original reads original, 50"] = scorecard.few_label_accuracy(
        slot_order[0], train_labels, slot_order[1], test_labels, picks[50])
    for n in picks:
        row[f"original reads original, {n}"] = scorecard.few_label_accuracy(o_train, train_labels, o_test, test_labels, picks[n])
        row[f"original reads new speaker, {n}"] = scorecard.few_label_accuracy(o_train, train_labels, n_test, test_labels, picks[n])
        row[f"new speaker on its own, {n}"] = scorecard.few_label_accuracy(n_train, train_labels, n_test, test_labels, picks[n])
    results[recipe] = row
    print(f"Scored {recipe} ({time.time() - start_all:.0f} s since start)")

print("\nResults on the 10,000 test images (accuracies: model B with 50 / 1,000 labels):\n")
print(f"{'':<34}{'alphabet':>12}{'flat':>12}")
for key in results["alphabet"]:
    cells = [results[r].get(key) for r in ("alphabet", "flat")]
    fmt = (lambda v: f"{v:.4f}") if "error" in key else (lambda v: f"{v:.1%}")
    print(f"{key:<34}" + "".join(("-" if v is None else fmt(v)).rjust(12) for v in cells))

# Step 4: verdict against the bar fixed before running (alphabet, 50 labels).
r = results["alphabet"]
lost = r["original reads original, 50"] - r["original reads new speaker, 50"]
quality = r["new speaker on its own, 50"] - r["original reads original, 50"]
print("\nVerdict (alphabet, 50 labels):")
print(f"  Same symbols: {r['same symbols']:.1%} (bar: >= 70%)  {'PASS' if r['same symbols'] >= 0.70 else 'FAIL'}")
print(f"  Understood by others: model B loses {lost:+.1%} reading the new speaker (bar: <= 2 points)  "
      f"{'PASS' if lost <= 0.02 else 'FAIL'}")
print(f"  Quality kept: new speaker {quality:+.1%} vs original (bar: within 1 point)  "
      f"{'PASS' if abs(quality) <= 0.01 or quality > 0 else 'FAIL'}")
print(f"\nTotal time: {(time.time() - start_all) / 60:.1f} min")
