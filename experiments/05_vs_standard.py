"""Experiment 5: experiment 4's VQ-VAE words vs standard methods.

Same test images and same random label picks as experiment 4. Compares:
  - few-label accuracy: raw pixels vs PCA (49 numbers) vs our 49 words
  - rebuild quality: PCA vs our words
  - storage: gzip, PNG, JPEG (quality 50/75/90), PCA, our words

Needs experiment 4 to have run first (it saves model A to data/models/).
"""
import io, sys, gzip
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fam.images import load_fashion_mnist
from fam.image_vqvae import ImageVQVAE, words_of

trX, trY = load_fashion_mnist("train"); teX, teY = load_fashion_mnist("t10k")
m = ImageVQVAE(); m.load_state_dict(torch.load(Path(__file__).resolve().parent.parent / "data" / "models" / "fashion_vqvae.pt")); m.eval()
trW, teW = words_of(m, trX), words_of(m, teX)

# Standard 1: PCA, the classic way to compress data into a few numbers (49, same as our 49 words).
pca = PCA(n_components=49, random_state=0).fit(trX.reshape(-1, 784) / 255.0)
trP, teP = pca.transform(trX.reshape(-1, 784) / 255.0), pca.transform(teX.reshape(-1, 784) / 255.0)
# Rebuild quality on test images
rebuilt_pca = pca.inverse_transform(teP)
pca_err = np.mean((rebuilt_pca - teX.reshape(-1, 784) / 255.0) ** 2)
with torch.no_grad():
    vq_err = 0.0
    for i in range(0, 10000, 1000):
        r = m.decode_words(torch.tensor(teW[i:i+1000], dtype=torch.long)).squeeze(1).numpy()
        vq_err += np.sum((r - teX[i:i+1000] / 255.0) ** 2)
    vq_err /= teX.size
print(f"rebuild error on test images: pca(49) {pca_err:.4f}   vq words(49) {vq_err:.4f}")

with torch.no_grad():
    vec = lambda w: m.codebook[torch.tensor(w, dtype=torch.long)].reshape(len(w), -1).numpy()
    trV, teV = vec(trW), vec(teW)
inputs = {"pixels": (trX.reshape(-1, 784) / 255.0, teX.reshape(-1, 784) / 255.0),
          "pca49": (trP, teP), "words": (trV, teV)}
rng = np.random.default_rng(42)
print(f"{'labels':>7}" + "".join(f"{k:>9}" for k in inputs))
for n in [50, 100, 300, 1000, 5000]:
    s = {k: [] for k in inputs}
    for _ in range(5):
        pick = rng.choice(60000, n, replace=False)
        for k, (a, b) in inputs.items():
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(a[pick], trY[pick])
            s[k].append(c.score(b, teY))
    print(f"{n:>7}" + "".join(f"{np.mean(v):>9.1%}" for v in s.values())
          + "   (spread: " + ", ".join(f"±{np.std(v) * 100:.1f}" for v in s.values()) + ")")

# Standard 2: image formats. Tile all 10k test images into one big picture (100x100 grid).
grid = teX.reshape(100, 100, 28, 28).transpose(0, 2, 1, 3).reshape(2800, 2800)
img = Image.fromarray(grid)
def size(fmt, **kw):
    b = io.BytesIO(); img.save(b, fmt, **kw); return b.tell()
png = size("PNG", optimize=True)
for q in (50, 75, 90):
    jb = io.BytesIO(); img.save(jb, "JPEG", quality=q); j = jb.getvalue()
    back = np.asarray(Image.open(io.BytesIO(j)), dtype=np.float32).reshape(100, 28, 100, 28).transpose(0, 2, 1, 3).reshape(10000, 784) / 255.0
    err = np.mean((back - teX.reshape(-1, 784) / 255.0) ** 2)
    print(f"JPEG q{q}: {len(j)/1024:.0f} KB, rebuild error {err:.4f}")
print(f"PNG (lossless): {png/1024:.0f} KB")
print(f"gzip raw: {len(gzip.compress(teX.tobytes()))/1024:.0f} KB")
print(f"words zipped: {len(gzip.compress(teW.tobytes()))/1024:.0f} KB + 16 KB codebook")
pca_bytes = 10000 * 49 * 1  # if PCA numbers were rounded to 1 byte each (generous to PCA)
print(f"PCA 49 numbers @1 byte each: {pca_bytes/1024:.0f} KB (+ {pca.components_.astype(np.float32).nbytes/1024:.0f} KB basis)")
