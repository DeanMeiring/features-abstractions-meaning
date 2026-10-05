"""Experiment 16: which of our past results are real, and which could be luck?

Many results were decided by 1-2 points while scores spread by about +-3
between random picks of labelled images. This re-checks the key claims with
95% confidence intervals on PAIRED differences (both inputs tested on the same
20 picks of 50 labels as before), using the saved models; nothing is retrained.

    CLEAR PASS         the whole interval is on the right side of the bar
    CLEAR FAIL         the whole interval is on the wrong side
    TOO CLOSE TO CALL  the interval crosses the bar: more data needed

Needs experiments 4, 10, 14 and 15 first.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch
from sklearn.decomposition import PCA

from fam import image_vqvae, scorecard, set_vqvae
from fam.images import load_fashion_mnist

MODELS = ROOT / "data" / "models"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
rng = np.random.default_rng(42)
picks = [rng.choice(len(train_images), size=50, replace=False) for _ in range(20)]  # same 20 picks as before


def scores(train_x, test_x):
    return scorecard.few_label_scores(train_x, train_labels, test_x, test_labels, picks)


def table_vectors(table, words, sort=False):
    if sort:
        words = np.sort(words, axis=1)
    with torch.no_grad():
        return table[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


def load_set(name, alphabet):
    model = set_vqvae.SetVQVAE(alphabet=alphabet)
    model.load_state_dict(torch.load(MODELS / name))
    return model.eval()


def set_inputs(model, sort=False, writer=None):
    """Train and test inputs for model B from a set model's messages (optionally written by another speaker)."""
    writer = writer or model
    return (table_vectors(model.vectors(), set_vqvae.words_of(model, train_images), sort),
            table_vectors(model.vectors(), set_vqvae.words_of(writer, test_images), sort))


# The inputs the claims compare.
pixels = (train_images.reshape(-1, 784) / 255.0, test_images.reshape(-1, 784) / 255.0)
thumbnail = tuple(im.reshape(len(im), 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(im), -1) for im in (train_images, test_images))
pca = PCA(n_components=8, random_state=0).fit(pixels[0])
tr, te = pca.transform(pixels[0]), pca.transform(pixels[1])
low, high = tr.min(axis=0), tr.max(axis=0)
pca8 = tuple(np.round(np.clip((c - low) / (high - low), 0, 1) * 255) for c in (tr, te))

grid = image_vqvae.ImageVQVAE()
grid.load_state_dict(torch.load(MODELS / "fashion_vqvae.pt"))
grid.eval()
grid_words = tuple(table_vectors(grid.codebook, image_vqvae.words_of(grid, im)) for im in (train_images, test_images))

flat = {s: load_set(f, False) for s, f in [(42, "fashion_set8.pt"), (7, "fashion_set8_seed7.pt")]}
alpha = {s: load_set(f, True) for s, f in [(42, "fashion_alpha8.pt"), (7, "fashion_alpha8_seed7.pt")]}
speaker = load_set("fashion_alpha8_speaker7.pt", True)

s = {
    "pixels": scores(*pixels),
    "thumbnail": scores(*thumbnail),
    "PCA-8": scores(*pca8),
    "grid words": scores(*grid_words),
    "set words": scores(*set_inputs(flat[42])),
    "flat (2 seeds)": (scores(*set_inputs(flat[42])) + scores(*set_inputs(flat[7]))) / 2,
    "alphabet (2 seeds)": (scores(*set_inputs(alpha[42])) + scores(*set_inputs(alpha[7]))) / 2,
    "original reads original": scores(*set_inputs(alpha[42], sort=True)),
    "original reads new speaker": scores(*set_inputs(alpha[42], sort=True, writer=speaker)),
}

# (claim, experiment, paired difference per pick, bar, higher is better)
claims = [
    ("grid words beat pixels", "4", s["grid words"] - s["pixels"], 0.0, True),
    ("grid words beat the 7x7 thumbnail by 5 points", "7", s["grid words"] - s["thumbnail"], 0.05, True),
    ("8 set words reach the thumbnail's accuracy", "10", s["set words"] - s["thumbnail"], 0.0, True),
    ("8 set words beat PCA-8 by 2 points", "10", s["set words"] - s["PCA-8"], 0.02, True),
    ("alphabet within 1 point of flat words", "14", s["alphabet (2 seeds)"] - s["flat (2 seeds)"], -0.01, True),
    ("model B loses at most 2 points reading a new speaker", "15",
     s["original reads original"] - s["original reads new speaker"], 0.02, False),
]

print("Paired differences at 50 labels, 20 picks, with 95% confidence intervals:\n")
print(f"{'exp':>4}  {'claim':<52}{'difference':>11}  {'95% interval':>18}  {'bar':>6}  verdict")
for claim, exp, diff, bar, higher in claims:
    mean, lo, hi = scorecard.interval(diff)
    print(f"{exp:>4}  {claim:<52}{mean * 100:>+10.1f}  [{lo * 100:+6.1f}, {hi * 100:+6.1f}]  {bar * 100:>+6.1f}  "
          f"{scorecard.verdict(lo, hi, bar, higher)}")
