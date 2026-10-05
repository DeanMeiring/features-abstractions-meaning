"""Experiment 6: words of words. Do level-2 words help a model learn from few labels?

Level 1 (experiment 4): each image is a 7x7 grid of 49 words, each a 4x4 patch.
Level 2 (this one):     a second VQ-VAE reads only the level-1 words and writes
                        each image as a 4x4 grid of 16 words, each summarising
                        a 3x3 block of level-1 words. No labels, no pixels.

Model B: logistic regression, trained on 50 to 1,000 labelled images, from:
    pixels   784 raw brightnesses
    level 1  49 words x 16-dim vectors = 784 numbers
    level 2  16 words x 32-dim vectors = 512 numbers
averaged over 20 random picks of labelled images (5 were too noisy before).

Success bar, fixed in the README before running:
    keep level 2 (MDL): at 50 and 100 labels, level 2 is at most 1 point below level 1
    helps more:         at 50 labels, level 2 beats level 1 by at least 2 points

Needs experiment 4 to have run first (it saves the level-1 model to data/models/).
"""

import gzip
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam import image_vqvae, word_vqvae
from fam.images import CLASSES, load_fashion_mnist

LABEL_SIZES = [50, 100, 300, 1000]
REPEATS = 20
LEVEL1_FILE = ROOT / "data" / "models" / "fashion_vqvae.pt"
LEVEL2_FILE = ROOT / "data" / "models" / "fashion_words2.pt"
PICTURE = ROOT / "results" / "06_words_of_words.png"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")

# Step 1: level-1 words from experiment 4's model.
level1 = image_vqvae.ImageVQVAE()
level1.load_state_dict(torch.load(LEVEL1_FILE))
level1.eval()
train_w1 = image_vqvae.words_of(level1, train_images)
test_w1 = image_vqvae.words_of(level1, test_images)

# Step 2: level 2 learns its vocabulary from the level-1 word grids (cached).
if LEVEL2_FILE.exists():
    level2 = word_vqvae.WordVQVAE(level1.codebook)
    level2.load_state_dict(torch.load(LEVEL2_FILE))
    print(f"Loaded level-2 model from {LEVEL2_FILE.name}")
else:
    print("Training level 2 on 60,000 level-1 word grids, no labels, no pixels...")
    start = time.time()
    level2 = word_vqvae.train(train_w1, level1.codebook)
    print(f"  took {time.time() - start:.0f} s")
    torch.save(level2.state_dict(), LEVEL2_FILE)
level2.eval()
train_w2 = word_vqvae.words_of(level2, train_w1)
test_w2 = word_vqvae.words_of(level2, test_w1)

# Step 3: how much survives the squeeze from 49 to 16 words, on unseen test images?
with torch.no_grad():
    guessed_w1 = level2.guess_level1(torch.tensor(test_w2, dtype=torch.long))
    from_level1 = level1.decode_words(torch.tensor(test_w1, dtype=torch.long)).squeeze(1).numpy()
    from_level2 = level1.decode_words(guessed_w1).squeeze(1).numpy()
pixels = test_images / 255.0
print(f"\nTest images use {len(np.unique(test_w2))}/256 level-2 words")
print(f"Level-1 words recovered from level 2: {(guessed_w1.numpy() == test_w1).mean():.0%}")
print(f"Image rebuild error: from level 1 {((from_level1 - pixels) ** 2).mean():.4f}, "
      f"from level 2 {((from_level2 - pixels) ** 2).mean():.4f}")


def vectors(codebook, words):
    """Look each word up in its codebook and flatten: (n, h, w) -> (n, h * w * size)."""
    with torch.no_grad():
        return codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


inputs = {
    "pixels": (train_images.reshape(len(train_images), -1) / 255.0, pixels.reshape(len(pixels), -1)),
    "level 1": (vectors(level1.codebook, train_w1), vectors(level1.codebook, test_w1)),
    "level 2": (vectors(level2.codebook, train_w2), vectors(level2.codebook, test_w2)),
}

# Step 4: model B learns to classify from only n labelled images.
print(f"\nModel B test accuracy (10 classes; guessing = 10%), mean ± spread over {REPEATS} picks:\n")
print(f"{'labels':>7}" + "".join(f"{name:>16}" for name in inputs))
rng = np.random.default_rng(42)
results = {}
for n in LABEL_SIZES:
    scores = {name: [] for name in inputs}
    for _ in range(REPEATS):
        pick = rng.choice(len(train_images), size=n, replace=False)
        for name, (inp_train, inp_test) in inputs.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(inp_train[pick], train_labels[pick])
            scores[name].append(clf.score(inp_test, test_labels))
    results[n] = {name: np.mean(s) for name, s in scores.items()}
    print(f"{n:>7}" + "".join(f"{np.mean(s):>10.1%} ±{np.std(s):4.1%}" for s in scores.values()))

# Step 5: storage for the 10,000 test images.
raw_zip = len(gzip.compress(test_images.tobytes()))
print("\nStorage for 10,000 test images (zipped):")
print(f"  pixels:  {raw_zip / 1024:5.0f} KB")
for name, words, codebook in [("level 1", test_w1, level1.codebook), ("level 2", test_w2, level2.codebook)]:
    size = len(gzip.compress(words.tobytes()))
    print(f"  {name}: {size / 1024:5.0f} KB  ({raw_zip / size:.1f}x smaller than pixels; "
          f"{words[0].size} bytes per image; codebook {codebook.detach().numpy().nbytes / 1024:.0f} KB)")

# Step 6: the verdict against the bar fixed before running.
gap50 = results[50]["level 2"] - results[50]["level 1"]
gap100 = results[100]["level 2"] - results[100]["level 1"]
print(f"\nLevel 2 minus level 1: {gap50:+.1%} at 50 labels, {gap100:+.1%} at 100 labels")
print(f"  Keep level 2 (MDL, both gaps >= -1 point): {'PASS' if min(gap50, gap100) >= -0.01 else 'FAIL'}")
print(f"  Helps more (gap at 50 >= +2 points):       {'PASS' if gap50 >= 0.02 else 'FAIL'}")

# Step 7: picture: originals, rebuilt from 49 level-1 words, rebuilt from 16 level-2 words.
fig, axes = plt.subplots(3, 10, figsize=(12, 4.2))
for i in range(10):
    axes[0, i].set_title(CLASSES[test_labels[i]], fontsize=8)
    for row, img in enumerate([test_images[i], from_level1[i], from_level2[i]]):
        axes[row, i].imshow(img, cmap="gray")
        axes[row, i].axis("off")
for row, text in enumerate(["original\n784 bytes", "level 1\n49 bytes", "level 2\n16 bytes"]):
    axes[row, 0].text(-8, 14, text, ha="right", va="center", fontsize=8)
fig.savefig(PICTURE, dpi=120, bbox_inches="tight")
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
