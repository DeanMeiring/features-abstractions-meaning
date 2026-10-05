"""Experiment 4: can words learned from images help a model learn from few labels?

Model A: an image VQ-VAE learns a 256-word vocabulary from the 60,000
         training images, without labels. Each image becomes 7x7 = 49 words.
Model B: logistic regression classifies clothing (10 classes), trained on
         only 50 to 5,000 labelled images, from either:
           pixels  the raw 784 pixel brightnesses
           words   the 49 words, looked up in the codebook (49 x 16 = 784 numbers)
         Both inputs have 784 numbers, so the only difference is what they mean.

Tested on the 10,000 test images neither model trained on. Also measures
storage and saves a picture of originals vs images rebuilt from their words.
"""

import gzip
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 45%)

import matplotlib

matplotlib.use("Agg")  # save pictures to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam.image_vqvae import ImageVQVAE, train, words_of
from fam.images import CLASSES, load_fashion_mnist

LABEL_SIZES = [50, 100, 300, 1000, 5000]
REPEATS = 5  # different random picks of labelled images, averaged
MODEL_FILE = ROOT / "data" / "models" / "fashion_vqvae.pt"
PICTURE = ROOT / "results" / "04_fashion_rebuilt.png"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")

# Step 1: model A learns the vocabulary (cached, since it takes a few minutes).
if MODEL_FILE.exists():
    model = ImageVQVAE()
    model.load_state_dict(torch.load(MODEL_FILE))
    print(f"Loaded model A from {MODEL_FILE.name}")
else:
    print("Training model A (VQ-VAE) on 60,000 images, no labels...")
    start = time.time()
    model = train(train_images, epochs=6)
    print(f"  took {time.time() - start:.0f} s")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
model.eval()

# Step 2: write every image as 49 words.
train_words = words_of(model, train_images)
test_words = words_of(model, test_images)
print(f"Test images use {len(np.unique(test_words))}/256 words")


def word_vectors(words):
    """Look each word up in the codebook: (n, 7, 7) -> (n, 784)."""
    with torch.no_grad():
        return model.codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


inputs = {
    "pixels": (train_images.reshape(len(train_images), -1) / 255.0,
               test_images.reshape(len(test_images), -1) / 255.0),
    "words": (word_vectors(train_words), word_vectors(test_words)),
}

# Step 3: model B learns to classify from only n labelled images.
print("\nModel B test accuracy (10 classes; guessing = 10%), averaged over 5 picks:\n")
print(f"{'labels':>7}  {'pixels':>7}  {'words':>7}")
rng = np.random.default_rng(42)
for n in LABEL_SIZES:
    scores = {name: [] for name in inputs}
    for _ in range(REPEATS):
        pick = rng.choice(len(train_images), size=n, replace=False)
        for name, (inp_train, inp_test) in inputs.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(inp_train[pick], train_labels[pick])
            scores[name].append(clf.score(inp_test, test_labels))
    print(f"{n:>7}  {np.mean(scores['pixels']):>7.1%}  {np.mean(scores['words']):>7.1%}")

# Step 4: storage for the 10,000 test images.
raw = test_images.tobytes()
words = test_words.tobytes()
codebook_kb = model.codebook.detach().numpy().nbytes / 1024
print("\nStorage for 10,000 test images:")
print(f"  raw pixels:         {len(raw) / 1024:7.0f} KB")
print(f"  raw pixels, zipped: {len(gzip.compress(raw)) / 1024:7.0f} KB")
print(f"  words (49 bytes):   {len(words) / 1024:7.0f} KB  "
      f"({len(gzip.compress(raw)) / len(words):.1f}x smaller than zipped pixels)")
print(f"  words, zipped:      {len(gzip.compress(words)) / 1024:7.0f} KB  "
      f"({len(gzip.compress(raw)) / len(gzip.compress(words)):.1f}x smaller than zipped pixels)")
print(f"  + codebook (shared, once): {codebook_kb:.0f} KB")

# Step 5: picture of 10 test images (top) and the same images rebuilt from words (bottom).
with torch.no_grad():
    rebuilt = model.decode_words(torch.tensor(test_words[:10], dtype=torch.long)).squeeze(1).numpy()
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i in range(10):
    axes[0, i].imshow(test_images[i], cmap="gray")
    axes[0, i].set_title(CLASSES[test_labels[i]], fontsize=8)
    axes[1, i].imshow(rebuilt[i], cmap="gray")
    for ax in axes[:, i]:
        ax.axis("off")
axes[0, 0].text(-8, 14, "original\n784 bytes", ha="right", va="center", fontsize=8)
axes[1, 0].text(-8, 14, "from words\n49 bytes", ha="right", va="center", fontsize=8)
PICTURE.parent.mkdir(exist_ok=True)
fig.savefig(PICTURE, dpi=120, bbox_inches="tight")
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
