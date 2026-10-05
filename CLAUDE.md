# CLAUDE.md

Context for Claude Code working in this repo.

## The vision

Programming languages like Python were designed for humans, but AI now writes most code. This project explores an AI-native language for ML features: a compact, precise way to define, store, and reuse features that LLMs and ML models read and write natively.

The goal is to give ML models and AI systems context over each other's features, i.e. what they have learned, not just raw data.

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
4. **A decoder LLM that reads the language:** an LLM trained specifically to unpack vectors into meaning (the same pattern as LLaVA-style models, where a small adapter feeds image vectors into an LLM). This makes the language usable by people and other AI systems.
5. **The payoff:** when both sides share the language, only the short message needs to be stored or moved, never the raw data. The cost of meaning is paid once, when the vocabulary is built, and every later message gets cheaper. This could change how data is moved and stored.

**Honest limit:** a language cannot create information. It moves the cost into the shared vocabulary, like a dictionary does. The hard problems are deciding when a new word is worth adding (MDL) and keeping word meanings stable as the vocabulary grows.

**What exists vs. what is new:** the pieces exist separately (entity embeddings, VQ-VAE codebooks, BPE vocabularies, emergent agent communication, semantic communication, vector-to-LLM adapters). Combining them into one shared, growing, composable language across models, with a dedicated reader LLM, appears largely unexplored.

**Example (telecom):** instead of hand-engineering features per cell tower (slow and expensive), learn one vector per tower that gives context to other towers and models; the decoder LLM can then explain what a tower's vector means.

### Core ideas

- **Feature library:** features are reusable building blocks, like packages on PyPI. Each library entry has:
  - **Definition:** written in the feature language and executable (compiles to pandas/SQL)
  - **Embedding:** a vector capturing what the feature means, for search and comparison
  - **Metadata:** inputs, time window, which models used it, measured performance, version
- **Recursive compression:** features used together over time (e.g. 10 features over a year) get compressed into a new, higher-level feature, which can itself be compressed again. Each level is a more abstract concept.
- **Shared context:** models build on each other's learned features instead of relearning from raw data.

## Current scope

- Solo developer, proof-of-concept stage
- All models are the developer's own, so there is no cross-organisation translation problem
- Models communicate through the central library only (no machine-to-machine yet)
- Use public datasets only, never proprietary or employer data
- **Parallel build with a friend:** a friend working in telecom is building a related system (learned vectors per cell tower) on their side. Started 2026-10-05, aiming for something working in about 4 months (~2027-02). The two projects **share ideas only, never data or code**; their employer's data must never enter this repo.

## The claim the PoC must prove

Features learned by one model, stored in the library in compact form, make a second model better or faster than starting from raw data.

### Success criteria (define before building)

Fix the bar in advance so results can't be rationalised afterwards. Candidate metrics for model B, library features vs. raw-only:

- **Accuracy:** AUC improvement over the raw-only baseline at the same data size
- **Data efficiency:** fraction of training data needed to reach the raw-only baseline AUC
- **Speed:** training time to reach the baseline AUC

Pick concrete thresholds (e.g. "reaches baseline AUC with ≥30% less data") and record them here before running the reuse test.

## PoC plan (telecom churn)

Datasets: IBM Telco Customer Churn (Kaggle, start here), Cell2Cell (Kaggle, larger), Orange / KDD Cup 2009.

1. **Baseline:** raw data → LightGBM → record AUC. Everything else must beat this number.
2. **Library v0:** simple feature library (files or SQLite). Definitions start as a minimal structured spec that compiles to pandas. Do not design the full language yet.
3. **First compression:** compress a group of related features into one learned representation (e.g. a small PyTorch autoencoder), then store it as a new library entry.
4. **Reuse test (the actual proof):** train model B on a related but different task (e.g. upgrade or plan-change prediction). Compare raw-only vs. library features (including the compressed ones) on accuracy, data needed, and training time.
5. **LLM layer:** give an LLM the library (definitions and metadata) plus a new task, and let it select and combine features. Later: train a decoder LLM (via a small adapter) that reads the vectors directly and explains what they mean.

Stack: Python, pandas/Polars, scikit-learn, LightGBM, PyTorch, SQLite.

### Known dataset limitations

- **Telco is a single snapshot with no timestamps.** It cannot test the time-respecting/leakage principle or "features used together over time" compression. Treat it as a warm-up for steps 1–3; move to Cell2Cell or another dataset with a time dimension early.
- **Telco has only a churn label.** The reuse test needs a second, related task on the same customers. Options: derive a label from existing columns (e.g. predict `Contract` type or `InternetService` uptake, removing that column and anything directly derived from it from the inputs), or use a dataset with multiple targets (KDD Cup 2009 has churn, appetency, and up-selling on the same customers).

## Principles

- **Prove before expanding:** keep each stage small and measurable, and always compare against the baseline.
- **Stopping rule for compression:** keep a new abstraction only if it improves results or makes the description shorter (MDL principle). Compression cannot create information; gains come from better abstraction and reuse.
- **Guard against leakage:** every feature must respect time (no future information). Each compression level is a new place for leakage to sneak in.
- **Version everything:** features drift and stale patterns must be detectable.
- **Language cold-start:** LLMs know Python from billions of lines, but this language has none. Keep it small and regular so the spec fits in a prompt, and make it translatable to and from Python/SQL so synthetic training data can be generated later.

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

## Working with the developer

The developer is learning the fundamentals while building. Explain the why behind design choices, keep steps small, and favour clear, simple code over clever code.

- **One step at a time:** build one feature, train and evaluate it, show the result, then move on. Don't offer menus of options or ask for permission between small steps.
- **Short output:** report the key result (e.g. the AUC) and a line or two on what it means, not long dumps.

## Data

Datasets are not committed (`data/` is gitignored). Download with `python scripts/download_data.py`, which verifies the file's SHA-256 hash.
