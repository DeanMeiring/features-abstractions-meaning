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
| 3 | VQ-VAE: each customer becomes **4 words** from a learned 256-word vocabulary (4 bytes) | words only 0.912 · rebuild error 0.02 (unseen customers: 0.024, so not memorised) · **3.6x smaller than zipped raw data** | The first working form of the "language": a shared dictionary (4 KB) plus a 4-byte message per customer. Lossy, but beats zip on size. Caveat: bigger network and 3x more training than experiment 2's autoencoder, so not a perfectly fair comparison. |
| 4 | Image VQ-VAE (conv encoder, 28→14→7) learns 256 words from 60,000 images (no labels); each image becomes a 7×7 grid of **49 words** (49 bytes vs 784). Model B = logistic regression on 50–5,000 labelled images; both inputs are 784 numbers (pixels vs 49 words × 16-dim codebook vectors) | 50 labels: words 62.0% vs pixels 58.9% (±4) · 100: 69.5% vs 66.2% (±2) · 300 and 1,000: tie · 5,000: 81.1% vs 79.2% · words zipped 395 KB + 16 KB codebook vs 4,281 KB zipped pixels (**10.8x smaller**) | First time words beat raw data: model B learns faster from what model A learned. Gain is modest (~3 points, partly within the spread between random picks). |
| 5 | Experiment 4's words vs standard methods: PCA (49 numbers), PNG, JPEG | Words beat PCA by 9–11 points at 50–100 labels (PCA 50.5% / 58.8%) · rebuild error words 0.0050 vs PCA 0.0121 · 10x smaller than PNG, 4x smaller than JPEG q50 (but JPEG keeps ~3x more detail, rebuild error 0.0016) | Beats PCA, the classic compression for features, and beats file formats on size. Against JPEG it's a different trade-off, not a strict win: logos and fine patterns blur. |
| 6 | Words of words: a second VQ-VAE reads only the level-1 words and writes each image as a 4×4 grid of **16 level-2 words** (each summarising a 3×3 block of level-1 words). Model B over 20 picks | 50 labels: pixels 57.4% · level 1 61.3% · level 2 59.5% · 100 labels: 67.9% · 70.5% · 67.8% · level 2 recovers 44% of level-1 words · **29x smaller than zipped pixels** (16 bytes per image) | **Failed the bar fixed in advance** (level 2 was 1.8 / 2.7 points below level 1 at 50 / 100 labels; the bar allowed 1). A 3x shorter message, but it lost meaning model B needed. With 20 picks, level 1's win over pixels holds (+3.9 / +2.6 points). |

All Telco AUCs are averaged over repeated cross-validation splits (15 for experiment 1, 10 for 2–3), so one lucky test split can't decide a result. In experiments 2–3, model B predicts a different task (long contract vs month-to-month) with the `Contract` column hidden from both models. Image accuracies are averaged over 5 random picks of labelled images per size, tested on the 10,000 test images.

Experiment 4, originals (top, 784 bytes) vs rebuilt from 49 words (bottom, 49 bytes):

![Fashion-MNIST originals vs rebuilt from words](results/04_fashion_rebuilt.png)

Experiment 6, originals (top) vs rebuilt from 49 level-1 words (middle) vs from 16 level-2 words (bottom):

![Originals vs level-1 vs level-2 rebuilds](results/06_words_of_words.png)

### What we've learned

- **On small, clean data, raw features win.** Telco is small and clean; LightGBM finds the patterns itself, even from 50 labels. The learned language matters more where raw data is big, messy, or hard to read directly (images helped; Telco didn't).
- **Discrete words beat continuous codes.** Same idea, but snapping to a vocabulary gave better accuracy and real compression.
- **Messier data gives the language more room.** Images compressed 10.8x (vs 3.6x on Telco), and words helped model B with few labels.
- **Current words are low-level.** Each image word describes a 4×4 patch ("edge here"), not a concept ("shoe sole"). The gain over pixels is modest (~3 points) and partly within noise.
- **Squeezing harder isn't the same as abstracting.** Level 2 was trained to rebuild the level-1 words, so it kept what the image *looks like* at lower resolution (blurrier shapes) rather than what it *is*. Shorter messages, but the lost detail was detail model B used. A rebuild objective may not be enough to make higher-level words more meaningful.

### Next steps

1. **Recursive compression on Fashion-MNIST:** build a second level of words from the first-level 7×7 word grid (words of words), and test whether higher-level words help model B more with few labels. Keep level 2 only if it pays for itself (MDL).

   **Success bar for experiment 6, fixed before running** (level 2: each image as a 4×4 grid = 16 words, each word summarising a 3×3 block of level-1 words; model B averaged over 20 random picks per label size):
   - *Keep level 2 (MDL pass):* at 50 and 100 labels, level-2 words score no more than 1 point below level-1 words, while the message shrinks from 49 to 16 bytes.
   - *Higher-level words help more:* at 50 labels, level-2 words beat level-1 words by at least 2 points.

   **Result: both failed** (−1.8 points at 50 labels, −2.7 at 100). See experiment 6.

   **Success bar for experiment 7, fixed before running** (same-size comparison: is the 29x compression real, or would any 16-byte summary do?): every input stored at 1 byte per number; model B over the same 20 picks as experiment 6.
   - *Level 2 (16 bytes):* at 50 labels, level-2 words beat both a 4×4 thumbnail and PCA-16 by at least 5 points.
   - *Level 1 (49 bytes):* at 50 labels, level-1 words beat both a 7×7 thumbnail and PCA-49 by at least 5 points.
2. **Data with a real time dimension,** so models can learn by predicting what comes next. Cell2Cell and KDD Cup 2009 turned out to be single snapshots without real sequences; the current candidate is the Telecom Italia Milan Big Data Challenge (telecom traffic per grid square over ~2 months).
3. **Later:** the feature library (not built yet) and the decoder LLM.

## Run it

```bash
# everything runs on CPU; fam/compute.py caps training at 30% of the machine (threads, and GPU memory if present)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # for CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/download_data.py   # data is downloaded, not committed
python experiments/01_num_addon_services.py
python experiments/02_autoencoder_reuse.py
python experiments/03_vqvae_words.py   # ~3.5 min on 2 CPUs
python experiments/04_fashion_vqvae.py # ~7 min on 2 CPUs; saves model A to data/models/
python experiments/05_vs_standard.py   # needs experiment 4 first
python experiments/06_words_of_words.py # needs experiment 4 first; ~6 min at 4 threads
```

## Layout

- `fam/compute.py`: caps every experiment at 30% of the machine's computing power (imported first)
- `fam/data.py`: loading Telco and the shared evaluation (fixed splits, fixed model settings)
- `fam/autoencoder.py`: autoencoder that compresses a customer into a few numbers
- `fam/vqvae.py`: VQ-VAE that writes a customer as words from a learned vocabulary
- `fam/images.py`: loading Fashion-MNIST
- `fam/image_vqvae.py`: image VQ-VAE that writes a picture as a 7×7 grid of words
- `fam/word_vqvae.py`: level-2 VQ-VAE that writes a 7×7 grid of level-1 words as a 4×4 grid of level-2 words
- `experiments/`: one script per experiment, numbered in order
- `results/`: saved pictures from experiments
- `scripts/download_data.py`: downloads and verifies the datasets
