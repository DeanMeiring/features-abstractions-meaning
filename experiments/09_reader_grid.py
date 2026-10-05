"""Experiment 9: is the problem the words, or the reader?

Experiment 8's predict model clearly learned what clothes look like (it
redraws a hidden half), but a linear reader with 50 labels did no better with
its words. Either the knowledge isn't in the words, or the reader can't read
it. So: keep the words fixed and swap the reader; and give the reader more labels.

Readers (model B):
    linear        straight-line classifier on the word vectors (as before)
    bag of words  linear, on counts of each of the 256 words, ignoring position
    k-NN          no training: label of the 5 most similar training images
    MLP           small neural network (one hidden layer of 256)

Inputs: 7x7 thumbnail, copy / control / predict words (experiment 8), and the
predict model's numbers before they're snapped to words ("predict slots").
Labels: 50 (20 picks), 1,000 (5 picks), all 60,000 (the ceiling test).

Success bar: fixed in the README before running. Needs experiment 8 first.
"""

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fam.compute import CPU_THREADS  # noqa: E402  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam.image_vqvae import ImageVQVAE, to_tensor, words_of
from fam.images import load_fashion_mnist

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # MLP on 50 labels never "converges"; that's fine

PICKS = {50: 20, 1000: 5, 60000: 1}
MODELS = ROOT / "data" / "models"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
start_all = time.time()


def load(name, wide):
    model = ImageVQVAE(wide=wide)
    model.load_state_dict(torch.load(MODELS / name))
    return model.eval()


models = {
    "copy": load("fashion_vqvae.pt", wide=False),
    "control": load("fashion_wide_copy.pt", wide=True),
    "predict": load("fashion_wide_predict.pt", wide=True),
}


@torch.no_grad()
def slots_of(model, images, batch_size=1000):
    """The encoder's raw numbers before snapping to words, flattened to 784."""
    return np.concatenate([
        model.slots(to_tensor(images[i:i + batch_size])).reshape(-1, 784).numpy()
        for i in range(0, len(images), batch_size)
    ])


def word_vectors(model, words):
    with torch.no_grad():
        return model.codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


def bag_of_words(words):
    """How often each of the 256 words appears in the image, wherever it is."""
    counts = np.zeros((len(words), 256), dtype=np.float32)
    rows = np.repeat(np.arange(len(words)), 49)
    np.add.at(counts, (rows, words.reshape(-1)), 1)
    return counts


def thumbnail(images):
    return images.reshape(len(images), 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(images), -1).astype(np.float32)


# Step 1: every input, for train and test images. (name -> (train, test))
inputs = {"thumbnail": (thumbnail(train_images), thumbnail(test_images))}
bags = {}
for name, model in models.items():
    w_train, w_test = words_of(model, train_images), words_of(model, test_images)
    inputs[f"{name} words"] = (word_vectors(model, w_train), word_vectors(model, w_test))
    bags[f"{name} words"] = (bag_of_words(w_train), bag_of_words(w_test))
inputs["predict slots"] = (slots_of(models["predict"], train_images), slots_of(models["predict"], test_images))
print(f"Inputs ready ({time.time() - start_all:.0f} s)")


# Step 2: the readers. Each returns a fresh, untrained model B.
def linear(n):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def knn(n):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=CPU_THREADS))


def mlp(n):
    # With many labels, stop when a held-out 10% stops improving; with 50 there's too little to hold out.
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=200,
                                                         early_stopping=n >= 1000, random_state=0))


grid = [("linear", linear, inputs), ("bag of words", linear, bags), ("k-NN", knn, inputs), ("MLP", mlp, inputs)]

# Step 3: every reader x input x label size. Same picks for every reader and input.
rng = np.random.default_rng(42)
picks = {n: [rng.choice(len(train_images), size=n, replace=False) for _ in range(r)] for n, r in PICKS.items()}
results = {}
for reader, make, table in grid:
    t = time.time()
    for name, (inp_train, inp_test) in table.items():
        for n, n_picks in picks.items():
            scores = [make(n).fit(inp_train[p], train_labels[p]).score(inp_test, test_labels) for p in n_picks]
            results[reader, name, n] = np.mean(scores)
    print(f"{reader} done ({time.time() - t:.0f} s)")

print("\nModel B test accuracy (10 classes; guessing = 10%):\n")
print(f"{'reader':<13}{'input':<15}" + "".join(f"{n:>9,}" for n in PICKS))
for reader, _, table in grid:
    for name in table:
        print(f"{reader:<13}{name:<15}" + "".join(f"{results[reader, name, n]:>9.1%}" for n in PICKS))
    print()


# Step 4: verdict against the bar fixed before running.
def gap(reader, a, b, n):
    return results[reader, a, n] - results[reader, b, n]


readers = [r for r, _, _ in grid]
ceiling = gap("linear", "predict words", "control words", 60000)
best50 = max(readers, key=lambda r: results[r, "predict words", 50])
best_gap50 = gap(best50, "predict words", "control words", 50)
any_gap = max(gap(r, "predict words", "control words", n) for r in readers for n in PICKS)
slot_reader = max(["linear", "k-NN", "MLP"], key=lambda r: results[r, "predict words", 50])
snap = gap(slot_reader, "predict slots", "predict words", 50)
rescue = max(
    (gap(r, f"{m} words", "thumbnail", 50), r, m)
    for r in ["linear", "k-NN", "MLP"] for m in models
)

print("Verdict:")
print(f"  predict minus control words: all labels, linear {ceiling:+.1%} | best reader at 50 ({best50}) {best_gap50:+.1%}")
print(f"  Reader problem (either >= +2 points):            {'YES' if max(ceiling, best_gap50) >= 0.02 else 'NO'}")
print(f"  Words problem (no reader/size reaches +2 points): {'YES' if any_gap < 0.02 else 'NO'}  (best {any_gap:+.1%})")
print(f"  Snap loses it ({slot_reader}: slots minus words at 50, >= +2): {snap:+.1%}  {'YES' if snap >= 0.02 else 'NO'}")
print(f"  Reader rescues the claim (words minus thumbnail at 50, >= +5): best {rescue[0]:+.1%} "
      f"({rescue[2]} words, {rescue[1]})  {'YES' if rescue[0] >= 0.05 else 'NO'}")
print(f"\nTotal time: {(time.time() - start_all) / 60:.1f} min")
