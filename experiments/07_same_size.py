"""Experiment 7: are our words better than simple summaries of the same size?

Experiment 6 made each image 16 bytes (29x smaller than zipped pixels). But
"smaller than zipped pixels" is an easy bar. The fair test is against other
summaries that also take 16 (or 49) bytes per image, 1 byte per number:

    16 bytes  level-2 words (4x4)  vs  4x4 thumbnail  vs  PCA-16
    49 bytes  level-1 words (7x7)  vs  7x7 thumbnail  vs  PCA-49

Thumbnail: average each block of pixels (a blurry, tiny version of the image).
PCA:       the n directions along which the 60,000 training images vary most;
           each image is stored as its n positions along them.

No neural network is trained here: the word models come from experiments 4
and 6. Model B uses the same 20 picks of labelled images as experiment 6.

Success bar, fixed in the README before running: at 50 labels, our words beat
both same-size alternatives by at least 5 points, at each size.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam import image_vqvae, word_vqvae
from fam.images import load_fashion_mnist

LABEL_SIZES = [50, 100, 300, 1000]
REPEATS = 20
MODELS = ROOT / "data" / "models"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
train_px = train_images.reshape(len(train_images), -1) / 255.0
test_px = test_images.reshape(len(test_images), -1) / 255.0

# Step 1: our words, from the models trained in experiments 4 and 6.
level1 = image_vqvae.ImageVQVAE()
level1.load_state_dict(torch.load(MODELS / "fashion_vqvae.pt"))
level1.eval()
level2 = word_vqvae.WordVQVAE(level1.codebook)
level2.load_state_dict(torch.load(MODELS / "fashion_words2.pt"))
level2.eval()
train_w1, test_w1 = image_vqvae.words_of(level1, train_images), image_vqvae.words_of(level1, test_images)
train_w2, test_w2 = word_vqvae.words_of(level2, train_w1), word_vqvae.words_of(level2, test_w1)


def vectors(codebook, words):
    """Look each word up in its codebook and flatten: (n, h, w) -> (n, h * w * size)."""
    with torch.no_grad():
        return codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


# Step 2: thumbnails. 28 / 7 = 4, so a 7x7 thumbnail averages 4x4 pixel blocks;
# a 4x4 thumbnail averages 7x7 blocks. Each average is rounded to 1 byte.
def thumbnail(images, side):
    block = 28 // side
    small = images.reshape(len(images), side, block, side, block).mean(axis=(2, 4))
    return np.round(small).astype(np.uint8)


def blow_up(small):
    """Back to 28x28 by repeating each thumbnail pixel (to measure what was lost)."""
    block = 28 // small.shape[1]
    return small.repeat(block, axis=1).repeat(block, axis=2) / 255.0


# Step 3: PCA, with each of the n numbers rounded to 1 byte so it gets the same space.
def pca_bytes(n):
    pca = PCA(n_components=n, random_state=0).fit(train_px)
    train_c, test_c = pca.transform(train_px), pca.transform(test_px)
    low, high = train_c.min(axis=0), train_c.max(axis=0)  # range learned from training images only

    def to_byte(c):
        return np.round(np.clip((c - low) / (high - low), 0, 1) * 255).astype(np.uint8)

    def rebuild(b):
        return pca.inverse_transform(b / 255.0 * (high - low) + low).reshape(-1, 28, 28)

    return to_byte(train_c), to_byte(test_c), rebuild


# Step 4: every contestant: (bytes per image, train input, test input, test images rebuilt).
with torch.no_grad():
    rebuilt_w1 = level1.decode_words(torch.tensor(test_w1, dtype=torch.long)).squeeze(1).numpy()
    guessed_w1 = level2.guess_level1(torch.tensor(test_w2, dtype=torch.long))
    rebuilt_w2 = level1.decode_words(guessed_w1).squeeze(1).numpy()

contestants = {}
for side in (4, 7):
    tr, te = thumbnail(train_images, side), thumbnail(test_images, side)
    contestants[f"thumb {side}x{side}"] = (side * side, tr.reshape(len(tr), -1), te.reshape(len(te), -1), blow_up(te))
for n in (16, 49):
    tr, te, rebuild = pca_bytes(n)
    contestants[f"PCA-{n}"] = (n, tr, te, rebuild(te))
contestants["level-2 words"] = (16, vectors(level2.codebook, train_w2), vectors(level2.codebook, test_w2), rebuilt_w2)
contestants["level-1 words"] = (49, vectors(level1.codebook, train_w1), vectors(level1.codebook, test_w1), rebuilt_w1)
order = ["thumb 4x4", "PCA-16", "level-2 words", "thumb 7x7", "PCA-49", "level-1 words"]

print("Rebuild error on test images (lower = keeps more of the picture):")
for name in order:
    size, _, _, rebuilt = contestants[name]
    print(f"  {size:>2} bytes  {name:<14} {((rebuilt - test_images / 255.0) ** 2).mean():.4f}")

# Step 5: model B on each contestant, same 20 picks per size as experiment 6.
print(f"\nModel B test accuracy (10 classes; guessing = 10%), mean over {REPEATS} picks:\n")
print(f"{'labels':>7}" + "".join(f"{name:>15}" for name in order))
rng = np.random.default_rng(42)
results = {}
for n in LABEL_SIZES:
    scores = {name: [] for name in order}
    for _ in range(REPEATS):
        pick = rng.choice(len(train_images), size=n, replace=False)
        for name in order:
            _, inp_train, inp_test, _ = contestants[name]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(inp_train[pick], train_labels[pick])
            scores[name].append(clf.score(inp_test, test_labels))
    results[n] = {name: np.mean(s) for name, s in scores.items()}
    print(f"{n:>7}" + "".join(f"{results[n][name]:>15.1%}" for name in order))

# Step 6: verdict against the bar fixed before running.
print("\nAt 50 labels, words minus the best same-size alternative (bar: +5 points):")
for words, rivals in [("level-2 words", ["thumb 4x4", "PCA-16"]), ("level-1 words", ["thumb 7x7", "PCA-49"])]:
    best = max(rivals, key=lambda r: results[50][r])
    gap = results[50][words] - results[50][best]
    print(f"  {words} vs {best}: {gap:+.1%}  {'PASS' if gap >= 0.05 else 'FAIL'}")
