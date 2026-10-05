# features-abstractions-meaning

An experiment in building an **AI-native language for ML features**. Models compress what they learn into short "words" from a shared, growing vocabulary, so other models (and a reader LLM) can reuse that meaning instead of relearning it from raw data.

```
  RAW DATA (huge, messy)          THE "NEW LANGUAGE"            READERS
  Entity A: many events   ──►    vector A  ┐                ┌─► other models
  Entity B: ...           ──►    vector B  ├─ library of ───┤   (use as context)
  Entity C: ...           ──►    vector C  ┘   "words"      └─► decoder LLM
```

Like English: a small alphabet, an unlimited vocabulary, and short messages that carry a lot of meaning because the dictionary is shared. See [CLAUDE.md](CLAUDE.md) for the full vision and plan.

**Status:** proof of concept on public data: IBM Telco Customer Churn (7,043 customers, 19 input columns) and Fashion-MNIST (70,000 28×28 clothing images).

## Findings so far

| # | Experiment | Result | Takeaway |
|---|---|---|---|
| 1 | Hand-made feature `num_addon_services` added to the churn model | AUC 0.8448 → 0.8446 (no change) | The feature carries real signal (5% churn at 6 add-ons vs 46% at 1), but the model already learns it from the raw columns. |
| 2 | Autoencoder (model A) squeezes each customer into 4 numbers; model B predicts a new task (long contract) | raw 0.937 · raw+codes 0.936 · codes only 0.904 | 4 learned numbers keep most of the meaning of 18 columns for a task model A never saw, but add nothing on top of raw data. Not smaller than zip. |
| 3 | VQ-VAE: each customer becomes **4 words** from a learned 256-word vocabulary (4 bytes) | words only 0.912 · rebuild error 0.02 (unseen customers: 0.024) · **3.6x smaller than zipped raw data** | The first working form of the "language": a shared dictionary (4 KB) plus a 4-byte message per customer. Lossy, but beats zip on size. |
| 4 | Image VQ-VAE learns 256 words from 60,000 images (no labels); each image becomes **49 words** (49 bytes vs 784). Model B classifies clothing from few labels | 50 labels: words 62.0% vs pixels 58.9% · 100 labels: 69.5% vs 66.2% · **10.8x smaller than zipped pixels** | First time words beat raw data: model B learns faster from what model A learned. Gain is modest (~3 points, partly within the ±2–4 point spread between random picks). |
| 5 | Experiment 4's words vs standard methods: PCA (49 numbers), PNG, JPEG | Words beat PCA by 9–11 points at 50–100 labels, with 2.4x better rebuild · 10x smaller than PNG, 4x smaller than JPEG q50 (but JPEG keeps ~3x more detail) | Beats the classic compression for features (PCA) and standard file formats on size. Lossy: logos and fine patterns blur. |

All Telco AUCs are averaged over repeated cross-validation splits (15 for experiment 1, 10 for 2–3), so one lucky test split can't decide a result. Image accuracies are averaged over 5 random picks of labelled images, tested on the 10,000 test images.

Experiment 4, originals (top, 784 bytes) vs rebuilt from 49 words (bottom, 49 bytes):

![Fashion-MNIST originals vs rebuilt from words](results/04_fashion_rebuilt.png)

### What we've learned

- **On easy data, raw features win.** Telco is small and clean; LightGBM finds the patterns itself, even from 50 labels. The learned language should matter most where raw data is too big, messy, or expensive to read directly.
- **Discrete words beat continuous codes.** Same idea, but snapping to a vocabulary gave better accuracy and real compression.
- **Messier data gives the language more room.** Images compressed 10.8x (vs 3.6x on Telco), and words helped model B with few labels.
- **Current words are low-level.** Each image word describes a 4×4 patch ("edge here"), not a concept ("shoe sole").
- **Next:** recursive compression (words of words) to get higher-level words; then data with a time dimension, so models can learn by predicting what comes next.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # for CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/download_data.py   # data is downloaded, not committed
python experiments/01_num_addon_services.py
python experiments/02_autoencoder_reuse.py
python experiments/03_vqvae_words.py   # ~3.5 min on 2 CPUs
python experiments/04_fashion_vqvae.py # ~7 min on 2 CPUs; saves model A to data/models/
python experiments/05_vs_standard.py   # needs experiment 4 first
```

## Layout

- `fam/data.py`: loading Telco and the shared evaluation (fixed splits, fixed model settings)
- `fam/autoencoder.py`: autoencoder that compresses a customer into a few numbers
- `fam/vqvae.py`: VQ-VAE that writes a customer as words from a learned vocabulary
- `fam/images.py`: loading Fashion-MNIST
- `fam/image_vqvae.py`: image VQ-VAE that writes a picture as a 7×7 grid of words
- `experiments/`: one script per experiment, numbered in order
- `results/`: saved pictures from experiments
- `scripts/download_data.py`: downloads and verifies the dataset
