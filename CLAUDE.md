# CLAUDE.md

Context for Claude Code working in this repo.

## The vision

Programming languages like Python were designed for humans, but AI now writes most code. This project explores an AI-native language for ML features: a compact, precise way to define, store, and reuse features that LLMs and ML models read and write natively.

The goal is to give ML models and AI systems context over each other's features, i.e. what they have learned, not just raw data.

**What it is for (clarified after experiments 1–16):** an automatic translator from **raw data a human can't read** (sensor streams, traffic, pixels) into short, meaningful words that any model can reuse. It replaces the slow human step of turning raw data into features, not tidy tables that a person has already summarised (Telco showed there's nothing left to gain there). It is meant for many connected sources ("points", e.g. cell towers or road sensors, each connected to others). The payoff could be better accuracy, or the same accuracy at much lower cost (less data moved, faster training, fewer labels); which of these is the goal is an **open question** (see "The claim the PoC must prove"). The closest analogy is a compiler's shared intermediate language (like LLVM IR): instead of engineering features for every data-model pair, translate everything once into one shared language that every model reads. Unlike a compiler it is learned and lossy on purpose, closer to a video codec that keeps what matters.

### The big picture: compressed vectors as a new language

```
  RAW DATA (huge, messy)          THE "NEW LANGUAGE"            READERS
  Entity A: many events   ──►    vector A  ┐                ┌─► other models
  Entity B: ...           ──►    vector B  ├─ library of ───┤   (use as context)
  Entity C: ...           ──►    vector C  ┘   "words"      └─► decoder LLM
                                     │                          (unpacks vectors
                          compose ───┘                           into meaning)
                          (words of words)
```

1. **Compress:** an encoder squeezes raw data (a customer, a cell tower, an image) into a small learned vector. This is the middle of an autoencoder, trained by rebuilding the input or, more powerfully, by predicting what comes next.
2. **Treat the middle as a language, not a bottleneck:** normally the compressed vector is private to the one decoder trained with it, then thrown away. Here the vectors become a shared, stable vocabulary that any model can write and read.
3. **A growing vocabulary from a small alphabet:** like English (26 letters, unlimited words) or Chinese (characters as units of meaning that combine into words). Base symbols combine into learned "words", and words combine into higher-level words (recursive compression). The vocabulary grows large over time, but each message stays short because one word carries a lot of meaning.
4. **A reader LLM that reads the language:** an LLM trained specifically to unpack the words into meaning (the same pattern as LLaVA-style models, where a small adapter feeds image vectors into an LLM). This makes the language usable by people and other AI systems. **The reader replaces a hand-written feature code language** (the original plan had definitions compiling to pandas/SQL). It must stand on measured meaning: each word has a computed **word card** (where it appears, what it draws or predicts, how stable it is), and the LLM explains the cards rather than inventing meaning.
5. **The payoff:** when both sides share the language, only the short message needs to be stored or moved, never the raw data. The cost of meaning is paid once, when the vocabulary is built, and every later message gets cheaper. This could change how data is moved and stored.

**Honest limit:** a language cannot create information. It moves the cost into the shared vocabulary, like a dictionary does. The hard problems are deciding when a new word is worth adding (MDL) and keeping word meanings stable as the vocabulary grows.

**What exists vs. what is new:** the pieces exist separately (entity embeddings, VQ-VAE codebooks, BPE vocabularies, emergent agent communication, semantic communication, vector-to-LLM adapters). Combining them into one shared, growing, composable language across models, with a dedicated reader LLM, appears largely unexplored.

**Example (telecom):** instead of hand-engineering features per cell tower (slow and expensive), learn one vector per tower that gives context to other towers and models; the decoder LLM can then explain what a tower's vector means.

### Core ideas

- **One recipe, one dictionary per kind of data.** The recipe is the fixed method: build a dictionary from lots of raw data → check it on the scorecard → freeze and version it → onboard new speakers → use the messages. Each kind of data that shares patterns (clothing images, road traffic, tower load) gets its own dictionary; one per sensor would defeat sharing. The recipe's steps are fixed; its settings (words per message, window length) are tuned per dictionary.
- **Feature library:** reusable building blocks, like packages on PyPI. Each library entry has:
  - **Dictionary:** the frozen symbols (one vector each) plus the drawer that turns symbols back into data, with a version
  - **Word cards:** computed, checkable meaning for each symbol (replaces the old executable definition)
  - **Messages:** each item's words (and, for data over time, which point and when)
  - **Metadata:** data type, scorecard results, which models speak it, version history
- **Onboarding new speakers:** a new point or model doesn't reinvent the language; it learns the frozen dictionary (experiment 15), ideally with a small phrasebook of example messages.
- **Recursive compression:** features used together over time (e.g. 10 features over a year) get compressed into a new, higher-level feature, which can itself be compressed again. Each level is a more abstract concept. (Paused: experiment 6's attempt on meaningless grid words failed.)
- **Shared context:** models and connected points build on each other's learned words instead of relearning from raw data.

## Current scope

- Solo developer, proof-of-concept stage
- All models are the developer's own, so there is no cross-organisation translation problem
- Models communicate through the central library only (no machine-to-machine yet)
- Use public datasets only, never proprietary or employer data
- **Hardware:** a Windows laptop, CPU only (Intel Core Ultra 7 265U, 14 threads, 15.5 GB RAM), under the 30% compute limit (4 threads). Keep experiments to minutes. A second laptop with an NVIDIA RTX 3050/3060 is available later for GPU work (the reader LLM); its GPU-memory limit is still to be decided.
- The feature library is planned but not built yet; so far model A's words are saved under `data/models/` and read directly by model B.
- **Parallel build with a friend:** a friend working in telecom is building a related system (learned vectors per cell tower) on their side. Started 2026-10-05, aiming for something working in about 4 months (~2027-02). The two projects **share ideas only, never data or code**; their employer's data must never enter this repo.

## The claim the PoC must prove

Features learned by one model, stored in the library in compact form, make a second model better or faster than starting from raw data.

**Open question: better, or cheaper?** After experiment 10 the developer considered narrowing the claim to efficiency (the same accuracy as raw data at much lower cost: data moved, training time, labels), since large models already lead on accuracy. As of 2026-10-05 it's undecided ("too early to say"). It was raised after modest accuracy results, so if it's adopted later, record it as a dated change, and don't re-judge past experiments against it. So far neither an accuracy win at a scale that matters nor an efficiency win has been shown.

**Open question: a bar for the PoC as a whole.** Per-experiment bars never say when the PoC as a whole is proven or should stop, which is how experiments 6–16 drifted. As of 2026-10-05 this is deliberately left open, because it depends on the better-vs-cheaper question above, which the developer is still exploring (it may lean either way, or stay neutral). It still needs: what result, on non-toy data (PEMS-BAY or telecom), counts as "proven"; by what date (the ~2027-02 target alongside the friend's build is the natural one); and what counts as "stop or rethink". Until it's set, revisit this question whenever a non-toy experiment finishes, so the open bar doesn't become a reason to drift.

### Success criteria (define before building)

Fix the bar in the README before each experiment, so results can't be rationalised afterwards. Candidate metrics for model B, words vs. raw-only:

- **Accuracy:** improvement over the raw-only baseline at the same data size
- **Data efficiency:** labels or data needed to reach the raw-only baseline
- **Cost:** training time, and data stored or moved, at the same accuracy (with the one-off cost of building the dictionary reported separately)

Since experiment 16, a bar only counts as passed if the **whole 95% confidence interval** clears it (`fam/scorecard.py`); otherwise it's "too close to call".

## Plan

**Where it stands (experiments 1–16, details in the README):** on Fashion-MNIST, learned words clearly beat raw pixels with few labels, self-contained set words (8 bytes) clearly beat a 49-byte thumbnail, and a new model can learn a frozen dictionary well enough to be mostly understood by others. All effects are small and on a toy dataset.

**Best current evidence, not yet confirmed (don't treat as settled):** words seem to gain meaning when they must stand alone (experiment 10's set words: purity 17% → 70%, and a clear but small +1.8-point win over a thumbnail; but they failed "words are units" at 50 labels, and the win over PCA-8 was too close to call in experiment 16), and a shared language seems more stable when new speakers learn a frozen dictionary than when it's reinvented (experiment 15: agreement 47–51% → 58–69%; but "same symbols" failed its 70% bar and "understood by others" was too close to call). Library v0 builds on these as working assumptions, to be revisited if they don't hold.

On PEMS-BAY road traffic (the dataset closest to the thesis: connected points over time), neighbours' raw data helped only ~2.5% and words kept a fraction of that; the likely bottleneck is the forecaster, which has not been tested directly.

**Next, in this order:**
1. **PEMS-BAY post-mortem (experiment 17):** can any reasonable forecaster (not one shared LightGBM, e.g. per-sensor models or a simple graph model) get a real gain from neighbours' raw data? This decides whether messages over time pay off, and so whether the library needs a time dimension. **If it fails** (no reasonable forecaster gets a real gain from neighbours' raw data), the "many connected points" part of the vision gets revisited, not just the time column dropped: write that consequence into the experiment's bar before running, so a fail can't be absorbed quietly.
2. **Library v0:** SQLite, end-to-end, on the best confirmed words (experiment 10's set words): dictionaries (frozen, versioned), word cards, messages (with point and time if step 1 says so), scorecards, and a simple query interface. Clunky but complete.
3. **Reader LLM:** first an off-the-shelf LLM reading word cards (no training), scored against true labels; only then train a small adapter (GPU laptop or a free GPU notebook).
4. **Telecom:** the same recipe on telecom data (Telecom Italia), for the link to the friend's work.

Datasets in use: **Fashion-MNIST** (developing the recipe; a toy, not evidence on its own) and **PEMS-BAY** (325 connected road sensors, speed every 5 minutes, Jan–Jun 2017; the main time-and-network dataset). Telco churn was the warm-up.

Stack: Python, pandas, scikit-learn, LightGBM, PyTorch, matplotlib; SQLite for the library; Hugging Face transformers/peft later for the reader LLM.

### Known dataset limitations

- **Telco is a single snapshot of human-made columns.** Nothing left for the language to add; kept only as the warm-up (experiments 1–3).
- **Fashion-MNIST is a toy.** Good for fast recipe development, but results there show the idea *can* work, not that it matters.
- **PEMS-BAY is aggregated speed per sensor**, not raw events, and public results (e.g. DCRNN) exist to compare against. Cell2Cell and KDD Cup 2009 are snapshots without real sequences; Telecom Italia (traffic per grid square over ~2 months) is the telecom candidate.

## Principles

- **Prove before expanding:** keep each stage small and measurable, and always compare against the baseline.
- **Stopping rule for compression:** keep a new abstraction only if it improves results or makes the description shorter (MDL principle). Compression cannot create information; gains come from better abstraction and reuse.
- **Guard against leakage:** every feature must respect time (no future information). Each compression level is a new place for leakage to sneak in.
- **Version everything:** features drift and stale patterns must be detectable.
- **Language cold-start:** LLMs know Python from billions of lines, but this language has none. Keep it small and regular so word cards fit in a prompt, and since the words are discrete they can be written as text tokens (e.g. `<w29>`), so a text LLM can be taught to read them.
- **Measured meaning first:** a word's meaning is what its word card measures; the reader LLM explains it and never defines it.
- **No new encoder variant without a written gap.** Don't propose a new way of making words unless Library v0, the PEMS-BAY post-mortem or a later step reveals a specific gap the existing encoders can't fill, and write that gap down (in the README) before proposing the variant. Experiments 6–16 drifted into encoder variants while the library and the traffic question waited.

## Key references

- **DreamCoder / library learning (MIT):** closest blueprint. Learns reusable abstractions and compresses them into a growing library.
- **BPE tokenization:** the simplest working example of recursive compression into new symbols.
- **VQ-VAE / hierarchical VQ-VAE:** compressing data into discrete codes, in levels.
- **Feature stores (Feast, Tecton, Hopsworks) and Featuretools:** existing human-oriented feature libraries and primitives.
- **Platonic Representation Hypothesis (2024):** models tend to converge on similar representations, which supports the idea of a shared language.
- **Relative representations / model stitching:** translating between models' vector spaces (future, machine-to-machine).
- **TabLLM, CAAFE:** LLMs reading tabular data and LLMs generating features.
- **MDL principle:** when an abstraction is worth keeping.
- **Entity embeddings:** learning one vector per entity (store, listing, tower) instead of hand-engineering features.
- **LLaVA / vision-language adapters:** a small layer that feeds non-text vectors into an LLM so it can describe them; the blueprint for the decoder LLM.
- **Emergent communication:** AI agents inventing their own languages to talk to each other.
- **Semantic communication (6G research):** transmitting meaning instead of raw bits.
- **Self-supervised prediction (next-token, next-frame, JEPA):** learning representations by predicting what comes next.
- **DCRNN (Li et al., 2018) and graph neural networks:** forecasting connected sensors; DCRNN set the PEMS-BAY benchmark used here.
- **Chronos, TimesFM, VQ tokenizers for time series:** foundation models that already turn time series into tokens; relatives to learn from, not to compete with on accuracy.
- **Residual / product quantization:** building symbols from parts (the radical + detail alphabet of experiment 14).

## Working with the developer

The developer is learning the fundamentals while building. Explain the why behind design choices, keep steps small, and favour clear, simple code over clever code.

- **One step at a time:** build one feature, train and evaluate it, show the result, then move on. Don't offer menus of options or ask for permission between small steps.
- **Short output:** report the key result (e.g. the AUC) and a line or two on what it means, not long dumps.
- **Compute limit (30%):** training must never use more than 30% of the machine's computing power. Every experiment imports `fam.compute` first (before numpy/torch/lightgbm/sklearn), which caps CPU threads at 30% of logical cores for torch, LightGBM (`n_jobs`) and BLAS (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), and caps NVIDIA GPU memory at 30%. GPU compute can't be hard-capped, only its memory. RAM isn't hard-capped either, so keep batch sizes small enough to stay well under `MAX_RAM_GB`. New experiments and models must follow the same pattern (pass `n_jobs=CPU_THREADS` to anything that takes it).
- **README as the results log:** whenever committing and pushing new work, update `README.md` with the new findings (findings table, takeaways, run instructions) in the same commit.

## Data

Datasets are not committed (`data/` is gitignored). Download with `python scripts/download_data.py`, which verifies each file's SHA-256 hash. Datasets: IBM Telco churn, Fashion-MNIST, PEMS-BAY (with the road distances between sensors). Trained models are saved under `data/models/` (also not committed; rerun the experiment to recreate).

## Progress

Experiment results, takeaways and the next planned step live in the README's "Findings so far" section; read it at the start of a session to see where the project stands.
