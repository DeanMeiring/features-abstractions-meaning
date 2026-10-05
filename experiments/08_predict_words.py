"""Experiment 8: do words trained to PREDICT carry more meaning than words trained to COPY?

Experiments 6 and 7 showed words trained to copy the image are great
compressors, but for learning from few labels they barely beat a blurry 7x7
thumbnail. Copying rewards "what it looks like", not "what it is".

Here, during training, a random half of the image is hidden (in half of each
batch) and the model must still rebuild the whole image. To redraw a hidden
heel, the words have to "know" it is a shoe.

Contestants for model B (all words are 49 bytes per image, 7x7 grid):
    pixels          784 raw brightnesses
    thumb 7x7       blurry 7x7 thumbnail (the bar to beat, from experiment 7)
    copy words      experiment 4's words (narrow view, copy job)
    control words   wide view (every word sees the whole image), copy job
    predict words   wide view, predict-the-hidden-half job

Success bar, fixed in the README before running, at 50 labels:
    main:  predict words beat the 7x7 thumbnail by at least 5 points
    cause: predict words beat the control words by at least 2 points

Needs experiment 4 to have run first (for the copy words).
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fam.image_vqvae import ImageVQVAE, hide_random_halves, to_tensor, train, words_of
from fam.images import CLASSES, load_fashion_mnist

LABEL_SIZES = [50, 100, 300, 1000]
REPEATS = 20
MODELS = ROOT / "data" / "models"
PICTURE = ROOT / "results" / "08_predict_words.png"

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")


def load_or_train(name, wide, hide_halves):
    """Train a word model once and cache it, since training takes a few minutes."""
    path = MODELS / name
    if path.exists():
        model = ImageVQVAE(wide=wide)
        model.load_state_dict(torch.load(path))
        print(f"Loaded {name}")
    else:
        print(f"Training {name} (wide={wide}, hide_halves={hide_halves}) on 60,000 images, no labels...")
        start = time.time()
        model = train(train_images, epochs=6, wide=wide, hide_halves=hide_halves)
        print(f"  took {time.time() - start:.0f} s")
        torch.save(model.state_dict(), path)
    return model.eval()


# Steps 1-2: the three word models (copy words come from experiment 4).
models = {
    "copy words": load_or_train("fashion_vqvae.pt", wide=False, hide_halves=False),
    "control words": load_or_train("fashion_wide_copy.pt", wide=True, hide_halves=False),
    "predict words": load_or_train("fashion_wide_predict.pt", wide=True, hide_halves=True),
}


def word_vectors(model, images):
    """Images -> 49 words -> their codebook vectors, flattened to 784 numbers."""
    words = words_of(model, images)
    with torch.no_grad():
        return model.codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


def thumbnail(images):
    """Average each 4x4 pixel block: a blurry 7x7 image, rounded to 1 byte per number."""
    return np.round(images.reshape(len(images), 7, 4, 7, 4).mean(axis=(2, 4))).reshape(len(images), -1)


inputs = {
    "pixels": (train_images.reshape(len(train_images), -1) / 255.0, test_images.reshape(len(test_images), -1) / 255.0),
    "thumb 7x7": (thumbnail(train_images), thumbnail(test_images)),
}
for name, model in models.items():
    inputs[name] = (word_vectors(model, train_images), word_vectors(model, test_images))

# Step 3: how well each model rebuilds whole test images, and the hidden half of hidden-half test images.
torch.manual_seed(0)
x = to_tensor(test_images)
x_half = hide_random_halves(torch.cat([x, x]))[:len(x)]  # first half of the doubled batch = every test image, half hidden
print("\nRebuild error on test images:   whole image   hidden half shown blank")
rebuilt = {}
with torch.no_grad():
    for name, model in models.items():
        full = model.decode_words(model.nearest_words(model.slots(x)))
        guess = model.decode_words(model.nearest_words(model.slots(x_half)))
        rebuilt[name] = guess.squeeze(1).numpy()
        print(f"  {name:<14} {((full - x) ** 2).mean():>20.4f} {((guess - x) ** 2).mean():>22.4f}")

# Step 4: model B, same 20 picks per size as experiments 6 and 7.
print(f"\nModel B test accuracy (10 classes; guessing = 10%), mean over {REPEATS} picks:\n")
print(f"{'labels':>7}" + "".join(f"{name:>15}" for name in inputs))
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
    print(f"{n:>7}" + "".join(f"{results[n][name]:>15.1%}" for name in inputs))

# Step 5: verdict against the bar fixed before running.
main = results[50]["predict words"] - results[50]["thumb 7x7"]
cause = results[50]["predict words"] - results[50]["control words"]
print("\nAt 50 labels:")
print(f"  main:  predict words minus thumbnail      {main:+.1%}  (bar +5.0%)  {'PASS' if main >= 0.05 else 'FAIL'}")
print(f"  cause: predict words minus control words  {cause:+.1%}  (bar +2.0%)  {'PASS' if cause >= 0.02 else 'FAIL'}")

# Step 6: picture: images with half hidden, and each model's guess at the whole image.
rows = [("shown\n(half hidden)", x_half.squeeze(1).numpy())] + [(name.replace(" ", "\n"), img) for name, img in rebuilt.items()]
fig, axes = plt.subplots(len(rows), 10, figsize=(12, 1.4 * len(rows)))
for i in range(10):
    axes[0, i].set_title(CLASSES[test_labels[i]], fontsize=8)
    for r, (_, imgs) in enumerate(rows):
        axes[r, i].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[r, i].axis("off")
for r, (label, _) in enumerate(rows):
    axes[r, 0].text(-8, 14, label, ha="right", va="center", fontsize=8)
fig.savefig(PICTURE, dpi=120, bbox_inches="tight")
print(f"\nSaved picture: {PICTURE.relative_to(ROOT)}")
