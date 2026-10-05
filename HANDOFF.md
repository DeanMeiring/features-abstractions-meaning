# Handoff: continuing on another machine

Written 2026-10-05 at the end of a long session on the first laptop, for picking the project up on the developer's second laptop (NVIDIA RTX 3050/3060, ~12 GB RAM, Intel i5). Read in this order: `CLAUDE.md` (vision, plan, rules), `README.md` (every experiment and its result), then this file (setup, plus context that isn't in the experiment log).

## 1. Setup on the new laptop

Explain each step briefly as you go; the developer is learning. Keep output short.

1. **Clone** `https://github.com/DeanMeiring/features-abstractions-meaning` (skip if already cloned) and read `CLAUDE.md` and `README.md` fully.
2. **Check the specs** and show them in a short table: OS, CPU model and cores/threads, RAM, free disk, Python version (3.11+), GPU (`nvidia-smi`: model and VRAM, or "none").
3. **Compute limit.** `fam/compute.py` caps CPU threads at `SHARE` (now 30%) of logical cores for torch, LightGBM and BLAS, and caps NVIDIA GPU memory at the same share. Every experiment imports it first.
   - **Ask the developer what GPU-memory limit to use on this laptop.** At 30%, a 4–6 GB laptop GPU leaves 1.2–1.8 GB, which is too small to fine-tune even a small LLM. Options discussed: a higher GPU-memory share for training runs (e.g. 70–80%) while keeping CPU at 30%, or keeping 30% and doing LLM training on a free GPU notebook (Kaggle / Colab, 16 GB). If the GPU share should differ from the CPU share, give `fam/compute.py` a separate setting for it, and update the compute-limit rule in CLAUDE.md.
   - Be honest about what can't be capped: GPU compute can't be hard-limited from PyTorch, only its memory; RAM isn't hard-capped (keep batches small).
   - Verify while something trains: `python scripts/monitor_cpu.py experiments/01_num_addon_services.py` should show CPU at or under the limit.
4. **Virtual environment:** `python -m venv .venv`, activate it, then install PyTorch first: the CUDA build from pytorch.org's selector if there's an NVIDIA GPU, else `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Then `pip install -r requirements.txt`.
5. **Data:** `python scripts/download_data.py` (Telco, Fashion-MNIST, PEMS-BAY; every file is hash-checked; `data/` is gitignored).
6. **Verify** against the README (small differences on another machine are fine; say if anything is far off):
   - `experiments/01_num_addon_services.py` → baseline AUC **0.8448**
   - `experiments/04_fashion_vqvae.py` → 50 labels: words **~62%** vs pixels **~58.9%** (retrains the image model)
   - `experiments/11_traffic_baseline.py` → LightGBM **1.46 / 1.95 / 2.54** mph at 15 / 30 / 60 min
   - The experiment code runs on CPU. Moving the models to the GPU is a later step, not part of setup.
7. **Saved models:** `data/models/` isn't in git, so later experiments need earlier ones rerun first (each script's docstring says which: e.g. 14 needs 10; 15 needs 10 and 14; 16 needs 4, 10, 14, 15).
8. Commit only what the setup changed on purpose (e.g. `fam/compute.py`, CLAUDE.md), with the README updated if anything there changed.

## 2. Where the project stands (one paragraph)

Seventeen experiments. On Fashion-MNIST (a toy, used to develop the recipe): learned words clearly beat raw pixels with few labels (+4.0 points), 8 self-contained "set words" (8 bytes) clearly beat a 49-byte thumbnail (+1.8), the model discovers a small alphabet of real concepts on its own (radicals like "trouser legs", "bag handle"), and a new model can learn a frozen dictionary well enough to be mostly understood by another (close, not proven). On PEMS-BAY road traffic: neighbours' raw data gives every forecaster only ~2.5–3.5% (experiment 17, a clear fail of the 5% bar), so "many connected points" is under review. Tidy tables (Telco) and words-of-words also didn't pay off. All wins so far are small; none is yet on data that matters.

## 3. Decisions made on 2026-10-05 (all recorded in CLAUDE.md)

- **The reader LLM replaces the hand-written feature code language.** Meaning is measured (word cards); the LLM explains it, never defines it.
- **One recipe, one dictionary per kind of data:** build → check (scorecard) → freeze and version → onboard new speakers → use.
- **Library entry** = dictionary (frozen symbols + drawer) + word cards + messages (with point and time as plain labels) + metadata.
- **Plan order:** ~~traffic post-mortem~~ (done, fail) → **Library v0** → reader LLM → telecom.
- **Bars are fixed in the README before every run, judged on the whole 95% confidence interval** (`fam/scorecard.py`); "too close to call" counts as not shown, and there are no reruns with tweaks.
- **Tripwire:** no new encoder variant unless a step reveals a specific gap, written down first. Experiments 6–16 drifted into encoder variants; external reviews (another AI) caught it.

## 4. Open questions (deliberately undecided)

- **Better, or cheaper?** Higher accuracy, or the same accuracy at much lower cost. The developer: "too early to say; it can go either way or stay neutral". The cheaper side rests on a theory, not yet measured: one cost spike to build the dictionary, then cheap reuse, which pays off past a break-even number of reuses.
- **A bar for the PoC as a whole:** must be set **before the first non-toy reuse test** (words vs raw on PEMS-BAY or telecom).
- **Where do connected points matter?** Maybe for new points with little history, missing data, or events spreading between points. Untested.
- **Candidate narrow question** (proposed, not chosen): "Can a brand-new point get good predictions quickly by learning the shared dictionary?" Alternative: measure the break-even of the "one spike, then cheap reuse" cost.

## 5. Context that isn't in the experiment log

- **Why the developer is doing this:** to get a real proof of concept without big-lab resources, and to work on it seriously, because they believe software and data engineering are moving to higher and higher levels (like binary → Python) and a learned language for data could be part of that. Treat this as a direction, not a claim; judge progress by small, testable steps.
- **Honest assessment given at the end of the session:** the idea is sensible and in line with where AI infrastructure is heading, but most pieces exist already (VQ tokenizers, Chronos/TimesFM, RAG, feature stores); the possibly distinctive part is a shared, frozen, versioned dictionary with onboarding, a library and a reader, which is mostly unbuilt. A realistic outcome is a well-documented PoC and write-up, not a breakthrough.
- **Analogies that helped:** an LLVM-style shared intermediate language between data and models (but learned and lossy, like a video codec); a Chinese-style alphabet of radicals; "a word is a vector everyone agreed on in advance".
- **Infrastructure notes:** Railway (the developer's Hobby plan) runs CPU only, no GPUs, so it's not useful for LLM training; free GPU notebooks (Kaggle, Colab) or the gaming laptop are. Never upload the friend's employer data anywhere.
- **Working style that worked:** before each experiment, explain the steps in plain terms, give a time estimate, fix the bar in the README and commit it, smoke-test with tiny settings, then run. After: report the key numbers in a short table, say what they mean in simple terms, update the README in the same commit, push. When editing CLAUDE.md, show the developer the diff before committing. The developer gets tired in long sessions; offer short summaries.

## 6. The next concrete step

**Library v0:** SQLite, end-to-end, on experiment 10's set words (`data/models/fashion_set8.pt`, so run experiment 10 first on the new machine). Store: the dictionary (frozen, versioned), its scorecard, a word card per symbol (where it appears, purity, stability, what it draws alone), and the messages for every image (with an item id; point and time as plain labels for data over time). Add a simple query interface ("which words mean trousers?", "all images containing word 29"). Clunky but complete. Fix the bar for it in the README first, and check the tripwire rule before reaching for any new encoder.
