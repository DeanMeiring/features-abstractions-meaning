"""Experiment 14: flat words vs a learned Chinese-style alphabet, on one scorecard.

Both write each image as an unordered set of 8 symbols, 1 byte each (fam/set_vqvae.py):
    flat      each symbol is 1 of 256 unrelated words (experiment 10)
    alphabet  each symbol = 1 of 16 radicals + 1 of 16 details (like a
              Chinese character from a radical and another part)
Each recipe is trained twice with different seeds, and every model goes
through the scorecard (fam/scorecard.py): meaning, compactness, units,
stability between the two seeds, and (alphabet) what the radicals alone carry.

Success bar, fixed in the README before running (50 labels, mean of both seeds):
    fair swap:         alphabet accuracy within 1 point of flat
    more stable:       alphabet stability beats flat by >= 10 points
    radicals carry it: radicals alone reach >= 90% of the full alphabet's accuracy
    usable language:   the better recipe's stability is >= 70%
Needs experiment 10 first (its model is flat seed 42).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import matplotlib

matplotlib.use("Agg")  # save pictures to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from fam import scorecard, set_vqvae
from fam.images import CLASSES, load_fashion_mnist

MODELS = ROOT / "data" / "models"
PICTURE = ROOT / "results" / "14_radicals.png"
RUNS = {  # (recipe, seed): saved model
    ("flat", 42): "fashion_set8.pt",  # experiment 10
    ("flat", 7): "fashion_set8_seed7.pt",
    ("alphabet", 42): "fashion_alpha8.pt",
    ("alphabet", 7): "fashion_alpha8_seed7.pt",
}
start_all = time.time()

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
rng = np.random.default_rng(42)  # same picks as experiments 6-10 for the 50-label size
picks = {n: [rng.choice(len(train_images), size=n, replace=False) for _ in range(r)] for n, r in {50: 20, 1000: 5}.items()}

# Step 1: train (or load) the four models.
models = {}
for (recipe, seed), name in RUNS.items():
    model = set_vqvae.SetVQVAE(alphabet=recipe == "alphabet")
    if (MODELS / name).exists():
        model.load_state_dict(torch.load(MODELS / name))
        print(f"Loaded {name}")
    else:
        print(f"Training {recipe}, seed {seed}...")
        t = time.time()
        model = set_vqvae.train(train_images, seed=seed, alphabet=recipe == "alphabet")
        print(f"  took {time.time() - t:.0f} s")
        torch.save(model.state_dict(), MODELS / name)
    models[recipe, seed] = model.eval()


# Step 2: the scorecard for each model.
def vectors(table, words):
    with torch.no_grad():
        return table[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


card, words_train = {}, {}
for (recipe, seed), model in models.items():
    w_train, w_test = set_vqvae.words_of(model, train_images), set_vqvae.words_of(model, test_images)
    words_train[recipe, seed] = w_train
    with torch.no_grad():
        table = model.vectors()
        rebuilt = model.decode_words(torch.tensor(w_test, dtype=torch.long)).squeeze(1).numpy()
    row = {
        "rebuild error": float(((rebuilt - test_images / 255.0) ** 2).mean()),
        "symbols used": len(np.unique(w_train)),
        "purity": scorecard.purity(w_train, train_labels),
    }
    v_train, v_test = vectors(table, w_train), vectors(table, w_test)
    b_train, b_test = scorecard.bag_of_words(w_train), scorecard.bag_of_words(w_test)
    for n in picks:
        row[f"in order, {n}"] = scorecard.few_label_accuracy(v_train, train_labels, v_test, test_labels, picks[n])
        row[f"bag of words, {n}"] = scorecard.few_label_accuracy(b_train, train_labels, b_test, test_labels, picks[n])
    if recipe == "alphabet":
        r_train, r_test = w_train // 16, w_test // 16
        row["radical purity"] = scorecard.purity(r_train, train_labels, vocab=16)
        for n in picks:
            row[f"radicals only, {n}"] = scorecard.few_label_accuracy(
                vectors(model.radicals, r_train), train_labels, vectors(model.radicals, r_test), test_labels, picks[n])
    card[recipe, seed] = row
    print(f"Scored {recipe}, seed {seed} ({time.time() - start_all:.0f} s since start)")

# Step 3: stability between the two seeds of each recipe.
stable = {recipe: scorecard.stability(words_train[recipe, 42], words_train[recipe, 7]) for recipe in ("flat", "alphabet")}
chance = {recipe: scorecard.stability_by_chance(words_train[recipe, 42], words_train[recipe, 7])
          for recipe in ("flat", "alphabet")}
radical_stable = scorecard.stability(words_train["alphabet", 42] // 16, words_train["alphabet", 7] // 16, vocab=16)
radical_chance = scorecard.stability_by_chance(words_train["alphabet", 42] // 16, words_train["alphabet", 7] // 16,
                                               vocab=16)

print("\nScorecard (8 symbols = 8 bytes per image; accuracies are model B on 50 / 1,000 labels):\n")
keys = ["rebuild error", "symbols used", "purity", "in order, 50", "in order, 1000", "bag of words, 50",
        "bag of words, 1000", "radical purity", "radicals only, 50", "radicals only, 1000"]
print(f"{'':<20}" + "".join(f"{r} {s:>2}".rjust(16) for r, s in models))
for key in keys:
    cells = []
    for run in models:
        value = card[run].get(key)
        cells.append("-" if value is None else f"{value:.4f}" if key == "rebuild error"
                     else f"{value:.0f}" if key == "symbols used" else f"{value:.1%}")
    print(f"{key:<20}" + "".join(c.rjust(16) for c in cells))
print(f"\nStability between seeds (chance level in brackets): flat {stable['flat']:.1%} ({chance['flat']:.1%}), "
      f"alphabet {stable['alphabet']:.1%} ({chance['alphabet']:.1%}), "
      f"radicals alone {radical_stable:.1%} ({radical_chance:.1%})")


# Step 4: verdict against the bar fixed before running (50 labels, mean of both seeds).
def mean(recipe, key):
    return np.mean([card[recipe, s][key] for s in (42, 7)])


flat_acc, alpha_acc = mean("flat", "in order, 50"), mean("alphabet", "in order, 50")
radical_share = mean("alphabet", "radicals only, 50") / alpha_acc
print("\nVerdict (50 labels, mean of both seeds):")
print(f"  Fair swap: alphabet {alpha_acc:.1%} vs flat {flat_acc:.1%} (bar: no more than 1 point below)  "
      f"{'PASS' if alpha_acc >= flat_acc - 0.01 else 'FAIL'}")
print(f"  More stable: alphabet minus flat {stable['alphabet'] - stable['flat']:+.1%} (bar: >= +10 points)  "
      f"{'PASS' if stable['alphabet'] - stable['flat'] >= 0.10 else 'FAIL'}")
print(f"  Radicals carry it: radicals alone reach {radical_share:.0%} of the alphabet's accuracy (bar: >= 90%)  "
      f"{'PASS' if radical_share >= 0.90 else 'FAIL'}")
print(f"  Usable language: best stability {max(stable.values()):.1%} (bar: >= 70%)  "
      f"{'PASS' if max(stable.values()) >= 0.70 else 'FAIL'}")

# Step 5: picture of the alphabet: each of the 16 radicals (seed 42) drawn alone,
# with the clothing class it appears in most.
model = models["alphabet", 42]
r_present = scorecard.presence(words_train["alphabet", 42] // 16, vocab=16).astype(np.float32)
hits = np.stack([r_present[train_labels == c].sum(axis=0) for c in range(10)], axis=1)
with torch.no_grad():
    alone = torch.sigmoid(model.drawer(model.radicals)).squeeze(1).numpy()
fig, axes = plt.subplots(2, 8, figsize=(12, 4.4), facecolor="#fcfcfb")
for r, ax in enumerate(axes.flat):
    ax.imshow(alone[r], cmap="gray", vmin=0, vmax=1)
    share = hits[r].max() / max(hits[r].sum(), 1)
    ax.set_title(f"radical {r}\n{share:.0%} {CLASSES[hits[r].argmax()]}", fontsize=8, color="#0b0b0b")
    ax.axis("off")
fig.suptitle("The learned alphabet: 16 radicals, each drawn alone, with the class it appears in most",
             fontsize=10, color="#0b0b0b")
fig.tight_layout()
fig.savefig(PICTURE, dpi=120, facecolor="#fcfcfb")
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
