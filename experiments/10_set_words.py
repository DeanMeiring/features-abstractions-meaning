"""Experiment 10: can words mean something on their own?

Experiment 9 showed our grid words only mean something at their grid
position: counting them (bag of words) lost ~35 points. Here each image is
an unordered SET of 8 words (8 bytes). Each word draws its own picture layer
and the layers are added, so order and position can't carry meaning (see
fam/set_vqvae.py).

Model B readers: linear on the 8 word vectors, bag of words (counts of each
of the 256 words), small neural net (MLP). Competitors: 7x7 thumbnail (49
bytes) and PCA-8 (8 bytes). Labels: 50 (20 picks), 1,000 (5), 60,000 (1).

Also measures "word purity": for each word, the share of images containing
it that belong to its most common class (a word in 90% trousers ~ "trouser").

Success bar: fixed in the README before running. Needs experiment 4 first
(for the grid-word purity comparison).
"""

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import matplotlib

matplotlib.use("Agg")  # save pictures to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam import image_vqvae, set_vqvae
from fam.images import CLASSES, load_fashion_mnist

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # MLP on 50 labels never "converges"; that's fine

PICKS = {50: 20, 1000: 5, 60000: 1}
MODELS = ROOT / "data" / "models"
MODEL_FILE = MODELS / "fashion_set8.pt"
PICTURE = ROOT / "results" / "10_set_words.png"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
start_all = time.time()

# Step 1: model A learns the 256-word vocabulary; each image becomes 8 unordered words.
if MODEL_FILE.exists():
    model = set_vqvae.SetVQVAE()
    model.load_state_dict(torch.load(MODEL_FILE))
    print(f"Loaded {MODEL_FILE.name}")
else:
    print("Training set model (8 words per image) on 60,000 images, no labels...")
    t = time.time()
    model = set_vqvae.train(train_images)
    print(f"  took {time.time() - t:.0f} s")
    torch.save(model.state_dict(), MODEL_FILE)
model.eval()
train_w, test_w = set_vqvae.words_of(model, train_images), set_vqvae.words_of(model, test_images)

with torch.no_grad():
    rebuilt = model.decode_words(torch.tensor(test_w, dtype=torch.long)).squeeze(1).numpy()
print(f"Test images use {len(np.unique(test_w))}/256 words; "
      f"rebuild error {((rebuilt - test_images / 255.0) ** 2).mean():.4f} (grid words, 49 bytes: 0.0049)")


# Step 2: word purity, for these set words and for experiment 4's grid words.
def purity(words, labels):
    """Usage-weighted share of each word's images that belong to its most common class."""
    n_words = words.reshape(len(words), -1)
    hits = np.zeros((256, 10))
    for c in range(10):
        present = np.zeros((int((labels == c).sum()), 256), dtype=bool)
        rows = np.repeat(np.arange(len(present)), n_words.shape[1])
        present[rows, n_words[labels == c].reshape(-1)] = True
        hits[:, c] = present.sum(axis=0)
    used = hits.sum(axis=1) > 0
    return (hits[used].max(axis=1).sum() / hits[used].sum()), hits


grid_model = image_vqvae.ImageVQVAE()
grid_model.load_state_dict(torch.load(MODELS / "fashion_vqvae.pt"))
grid_purity, _ = purity(image_vqvae.words_of(grid_model.eval(), train_images), train_labels)
set_purity, set_hits = purity(train_w, train_labels)
print(f"Word purity (10% = word says nothing about the class, 100% = one class only): "
      f"grid words {grid_purity:.0%}, set words {set_purity:.0%}")


# Step 3: inputs for model B.
def word_vectors(words):
    with torch.no_grad():
        return model.codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


def bag_of_words(words):
    counts = np.zeros((len(words), 256), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(len(words)), words.shape[1]), words.reshape(-1)), 1)
    return counts


def thumbnail(images):
    return images.reshape(len(images), 7, 4, 7, 4).mean(axis=(2, 4)).reshape(len(images), -1).astype(np.float32)


def pca_bytes(n):
    """PCA with n numbers, each rounded to 1 byte (same space as n words)."""
    train_px = train_images.reshape(len(train_images), -1) / 255.0
    test_px = test_images.reshape(len(test_images), -1) / 255.0
    pca = PCA(n_components=n, random_state=0).fit(train_px)
    tr, te = pca.transform(train_px), pca.transform(test_px)
    low, high = tr.min(axis=0), tr.max(axis=0)
    return [np.round(np.clip((c - low) / (high - low), 0, 1) * 255).astype(np.float32) for c in (tr, te)]


inputs = {
    "thumbnail (49 B)": (thumbnail(train_images), thumbnail(test_images)),
    "PCA-8 (8 B)": tuple(pca_bytes(8)),
    "set words (8 B)": (word_vectors(train_w), word_vectors(test_w)),
}
bags = {"set words (8 B)": (bag_of_words(train_w), bag_of_words(test_w))}


def linear(n):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def mlp(n):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=200,
                                                         early_stopping=n >= 1000, random_state=0))


grid = [("linear", linear, inputs), ("bag of words", linear, bags), ("MLP", mlp, inputs)]

# Step 4: model B, every reader x input x label size, same picks throughout.
rng = np.random.default_rng(42)
picks = {n: [rng.choice(len(train_images), size=n, replace=False) for _ in range(r)] for n, r in PICKS.items()}
results = {}
for reader, make, table in grid:
    for name, (inp_train, inp_test) in table.items():
        for n, n_picks in picks.items():
            results[reader, name, n] = np.mean(
                [make(n).fit(inp_train[p], train_labels[p]).score(inp_test, test_labels) for p in n_picks])

print("\nModel B test accuracy (10 classes; guessing = 10%):\n")
print(f"{'reader':<13}{'input':<18}" + "".join(f"{n:>9,}" for n in PICKS))
for reader, _, table in grid:
    for name in table:
        print(f"{reader:<13}{name:<18}" + "".join(f"{results[reader, name, n]:>9.1%}" for n in PICKS))
    print()

# Step 5: verdict against the bar fixed before running (all at 50 labels).
W = "set words (8 B)"
units = results["bag of words", W, 50] - results["linear", W, 50]
best_words = max(results[r, W, 50] for r in ["linear", "bag of words", "MLP"])
best_thumb = max(results[r, "thumbnail (49 B)", 50] for r in ["linear", "MLP"])
best_pca = max(results[r, "PCA-8 (8 B)", 50] for r in ["linear", "MLP"])
print("Verdict (50 labels):")
print(f"  Words are units: bag of words minus linear {units:+.1%} (bar: >= -2 points)  "
      f"{'PASS' if units >= -0.02 else 'FAIL'}")
print(f"  Words carry meaning: best set words {best_words:.1%} vs best thumbnail {best_thumb:.1%} (bar: >=)  "
      f"{'PASS' if best_words >= best_thumb else 'FAIL'}")
print(f"                       vs best PCA-8 {best_pca:.1%}: {best_words - best_pca:+.1%} (bar: >= +2 points)  "
      f"{'PASS' if best_words - best_pca >= 0.02 else 'FAIL'}")

# Step 6: picture. Top: originals vs rebuilt from 8 words.
# Bottom: the 10 most used words, each drawn ALONE, with the class it mostly appears in.
top = np.argsort(-set_hits.sum(axis=1))[:10]
with torch.no_grad():
    alone = torch.sigmoid(model.drawer(model.codebook[torch.tensor(top)])).squeeze(1).numpy()
fig, axes = plt.subplots(3, 10, figsize=(12, 4.6))
for i in range(10):
    axes[0, i].imshow(test_images[i], cmap="gray")
    axes[0, i].set_title(CLASSES[test_labels[i]], fontsize=8)
    axes[1, i].imshow(rebuilt[i], cmap="gray", vmin=0, vmax=1)
    h = set_hits[top[i]]
    axes[2, i].imshow(alone[i], cmap="gray", vmin=0, vmax=1)
    axes[2, i].set_title(f"word {top[i]}\n{h.max() / h.sum():.0%} {CLASSES[h.argmax()]}", fontsize=7)
    for ax in axes[:, i]:
        ax.axis("off")
for row, text in enumerate(["original\n784 bytes", "from 8 words\n8 bytes", "one word\nalone"]):
    axes[row, 0].text(-8, 14, text, ha="right", va="center", fontsize=8)
fig.savefig(PICTURE, dpi=120, bbox_inches="tight")
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
print(f"Total time: {(time.time() - start_all) / 60:.1f} min")
