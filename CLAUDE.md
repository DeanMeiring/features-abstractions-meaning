# CLAUDE.md

Context for Claude Code working in this repo.

## The vision

Programming languages like Python were designed for humans, but AI now writes most code. This project explores an AI-native language for ML features: a compact, precise way to define, store, and reuse features that LLMs and ML models read and write natively.

The goal is to give ML models and AI systems context over each other's features, i.e. what they have learned, not just raw data.

**What it is for (clarified after experiments 1–16):** an automatic translator from **raw data a human can't read** (sensor streams, traffic, pixels) into short, meaningful words that any model can reuse. It replaces the slow human step of turning raw data into features, not tidy tables that a person has already summarised (Telco showed there's nothing left to gain there). It is meant for many connected sources ("points", e.g. cell towers or road sensors, each connected to others) (under review after experiment 17, see Plan). The payoff could be better accuracy, or the same accuracy at much lower cost (less data moved, faster training, fewer labels); which of these is the goal is an **open question** (see "The claim the PoC must prove"). The closest analogy is a compiler's shared intermediate language (like LLVM IR): instead of engineering features for every data-model pair, translate everything once into one shared language that every model reads. Unlike a compiler it is learned and lossy on purpose, closer to a video codec that keeps what matters.

**Why the developer is doing this (2026-10-05):** without the infrastructure or resources of a large lab, the aim is a real proof of concept, worked on seriously. The belief behind it: software and data engineering are moving to higher and higher levels, with developers working above raw data the way programmers moved from binary to Python, and a learned language for data could be part of that. Accurately, it's a high-level **data** language (it describes data, it doesn't give instructions like a programming language), learned and lossy.

**The long-term picture (a direction, not a claim):** a model that needs data doesn't search, fetch, clean and study raw data every time; it calls the library, reads short pre-digested words fast and cheaply, and produces results, each with a confidence attached so it can fall back to raw data when the words aren't detailed enough. Relatives today: retrieval (RAG) over vector databases, feature stores, foundation models. The twist here is that the digest is a shared, versioned, learned language with measured meaning.

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
- **Onboarding new speakers:** a new model (a new encoder, e.g. for a different source or setup) doesn't reinvent the language; it learns the frozen dictionary (experiment 15), ideally with a small phrasebook of example messages. A new point with the same kind of data, using the same dictionary version (another tower or sensor), needs no onboarding: the frozen encoder writes its words directly.
- **Recursive compression:** features used together over time (e.g. 10 features over a year) get compressed into a new, higher-level feature, which can itself be compressed again. Each level is a more abstract concept. (Paused: experiment 6's attempt on meaningless grid words failed.)
- **Shared context:** models and connected points build on each other's learned words instead of relearning from raw data (connected points under review after experiment 17, see Plan).

## Current scope

- Solo developer, proof-of-concept stage
- All models are the developer's own, so there is no cross-organisation translation problem
- Models communicate through the central library only (no machine-to-machine yet)
- Use public datasets only, never proprietary or employer data
- **Hardware:** a Windows laptop, CPU only (Intel Core Ultra 7 265U, 14 threads, 15.5 GB RAM), under the 30% compute limit (4 threads). Keep experiments to minutes. A second laptop, set up on 2026-10-05 (Intel i5-11320H, 8 threads, 11.8 GB RAM, NVIDIA RTX 3050 with 4 GB), runs the same experiments under the same limit: 2 threads and a 3.5 GB RAM budget, which experiments 10, 14 and 15 exceed (about 4.1 GB peak; accepted by the developer for these scripts). Its GPU-memory limit stays at 30% (1.2 GB) for now, the developer's choice on 2026-10-05; that is too small to train an LLM, so it must be revisited before the reader LLM's adapter step.
- The feature library exists as v0 (`fam/library.py`, built by experiment 18 into `data/library_v0.db`) for the Fashion-MNIST set words only; the other experiments still read model A's words directly from `data/models/`.
- **Parallel build with a friend:** a friend working in telecom is building a related system (learned vectors per cell tower) on their side. Started 2026-10-05, aiming for something working in about 4 months (~2027-02). The two projects **share ideas only, never data or code**; their employer's data must never enter this repo.

## The claim the PoC must prove

Features learned by one model, stored in the library in compact form, make a second model better or faster than starting from raw data.

**Open question: better, or cheaper?** After experiment 10 the developer considered narrowing the claim to efficiency (the same accuracy as raw data at much lower cost: data moved, training time, labels), since large models already lead on accuracy. As of 2026-10-05 it's undecided ("too early to say"). It was raised after modest accuracy results, so if it's adopted later, record it as a dated change, and don't re-judge past experiments against it. So far neither an accuracy win at a scale that matters nor an efficiency win has been shown. The efficiency side rests on a cost shape that is so far only theory: one spike to build the dictionary, then each reuse is cheap. That only pays off past a break-even number of reuses (models, points, questions), there's an ongoing cost to translate new data, and new dictionary versions are new spikes; measuring the break-even is the test of it.

**Open question: a bar for the PoC as a whole.** Per-experiment bars never say when the PoC as a whole is proven or should stop, which is how experiments 6–16 drifted. As of 2026-10-05 this is deliberately left open, because it depends on the better-vs-cheaper question above, which the developer is still exploring (it may lean either way, or stay neutral). It still needs: what result, on non-toy data (PEMS-BAY or telecom), counts as "proven"; by what date (the ~2027-02 target alongside the friend's build is the natural one); and what counts as "stop or rethink". **Deadline:** it must be set before the first non-toy reuse test runs (words vs raw on PEMS-BAY or telecom, after Library v0), so the bar exists before any result that could "prove" the claim. Experiment 17 tests the forecaster, not the words, so it can go ahead first.

### Success criteria (define before building)

Fix the bar in the README before each experiment, so results can't be rationalised afterwards. Commit the bar on its own before running, so git shows the order (experiment 18's bar and result went into one commit, so its timing can't be shown). Candidate metrics for model B, words vs. raw-only:

- **Accuracy:** improvement over the raw-only baseline at the same data size
- **Data efficiency:** labels or data needed to reach the raw-only baseline
- **Cost:** training time, and data stored or moved, at the same accuracy (with the one-off cost of building the dictionary reported separately)

Since experiment 16, a bar only counts as passed if the **whole 95% confidence interval** clears it (`fam/scorecard.py`); otherwise it's "too close to call".

## Plan

**Where it stands (experiments 1–18, details in the README):** on Fashion-MNIST, learned grid words clearly beat raw pixels with few labels (+4.0 points on the first laptop, +3.9 [+3.0, +4.8] when retrained on the second). Self-contained set words (8 bytes) beat a 49-byte thumbnail on the first laptop's dictionary (+1.8 [+0.7, +3.0]), but on the dictionary retrained on the second laptop with the same seed it is too close to call (+0.9 [−0.4, +2.2]), so that win depends on which training is used and is **no longer a clear result**. A new model can learn a frozen dictionary well enough to be mostly understood by others. All effects are small and on a toy dataset.

**Best current evidence, not yet confirmed (don't treat as settled):** words seem to gain meaning when they must stand alone (experiment 10's set words: purity 17% → 70%, and a small win over a thumbnail that was clear on one training, +1.8, and too close to call on a retraining, +0.9; but they failed "words are units" at 50 labels, and the win over PCA-8 was too close to call in experiment 16 on both trainings), and a shared language seems more stable when new speakers learn a frozen dictionary than when it's reinvented (experiment 15: agreement 47–51% → 58–69%; but "same symbols" failed its 70% bar, and "understood by others" was too close to call on the first laptop and a clear pass on the second). Library v0 builds on these as working assumptions, to be revisited if they don't hold.

**Retraining is not copying (2026-10-05).** The same recipe and seed on a second machine gave a slightly different dictionary (set words 60.2% vs 61.3% at 50 labels). A dictionary is only the same dictionary if its file is copied, which is what the library's hash checks. A claim counts as settled only if it holds across trainings.

**Library v0 is built (experiment 18):** one SQLite file (`fam/library.py`) with the frozen, hashed dictionary, its speaker, the messages, a word card per symbol and the scorecard. All four bars passed, but three only check that storing and reading back lose nothing. The fourth: adding up the class shares on an image's word cards names its class 76.7% [75.9%, 77.5%] of the time with no model trained. That is counting on all 60,000 training labels, not a few-label result. The searchable file is 14x bigger than the words in it, which makes the "cheaper" story harder, not easier.

On PEMS-BAY road traffic (the dataset closest to the thesis: connected points over time), neighbours' raw data helped only ~2.5% and words kept a fraction of that (experiments 12–13). **Experiment 17 (post-mortem) ruled out the forecaster as the cause:** three forecaster types (shared LightGBM, linear per sensor, a mini-DCRNN graph network) all gained at most 3.5% [2.9%, 4.0%] from neighbours, a clear fail of the developer's 5% bar. A better forecaster per sensor mattered far more (graph network without neighbours 2.22 mph vs LightGBM 2.55; DCRNN 2.07).

**"Many connected points" is under review (2026-10-05, per experiment 17's bar).** On PEMS-BAY a point learns far more from its own history than from its neighbours. Neighbours may still matter where a point has little or no history of its own (a new sensor or tower), where its own data is missing, or where events spread between points; these are open questions, not assumptions, and the network idea doesn't shape the design until one of them is tested.

**Next, in this order:**
1. ~~**PEMS-BAY post-mortem (experiment 17)**~~ **Done: FAIL** (see above). The library doesn't need to be designed around neighbours; point and time are stored as plain labels on messages.
2. ~~**Library v0**~~ **Done** (experiment 18, see above): SQLite, end-to-end, on experiment 10's set words: dictionaries (frozen, versioned), word cards, messages (with point and time as plain labels), scorecards, and a simple query interface.
3. **Reader LLM (next):** first an off-the-shelf LLM reading word cards (no training), scored against true labels; only then train a small adapter (GPU laptop or a free GPU notebook).
4. **Telecom:** the same recipe on telecom data (Telecom Italia), for the link to the friend's work.

**Candidate narrow question for the PoC (proposed 2026-10-05, not decided):** "Can a model reading words reach good accuracy with far fewer labels or less history than one reading raw data?" It tests the dictionary itself, not the neighbours: the frozen encoder writes words for any new point without learning anything, and a point with no history at all could only be predicted from its neighbours, which is the network question experiment 17 just failed on traffic. (An earlier draft, "can a brand-new point get good predictions quickly by learning the dictionary?", was dropped for that reason: it would have brought "connected points" back under a new name.) A second candidate is measuring the break-even of the "one spike, then cheap reuse" cost shape.

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
- **Compute limit (30%):** training must never use more than 30% of the machine's computing power. Every experiment imports `fam.compute` first (before numpy/torch/lightgbm/sklearn), which caps CPU threads at 30% of logical cores for torch, LightGBM (`n_jobs`) and BLAS (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), and caps NVIDIA GPU memory at `GPU_SHARE` (also 30% for now; a separate setting so GPU memory can be raised for LLM training without raising CPU use). GPU compute can't be hard-capped, only its memory. RAM isn't hard-capped either, so keep batch sizes small enough to stay well under `MAX_RAM_GB`. New experiments and models must follow the same pattern (pass `n_jobs=CPU_THREADS` to anything that takes it).
- **README as the results log:** whenever committing and pushing new work, update `README.md` with the new findings (findings table, takeaways, run instructions) in the same commit.

## Data

Datasets are not committed (`data/` is gitignored). Download with `python scripts/download_data.py`, which verifies each file's SHA-256 hash. Datasets: IBM Telco churn, Fashion-MNIST, PEMS-BAY (with the road distances between sensors). Trained models are saved under `data/models/` (also not committed; rerun the experiment to recreate).

## Progress

Experiment results, takeaways and the next planned step live in the README's "Findings so far" section; read it at the start of a session to see where the project stands. `HANDOFF.md` has the setup steps for a new machine and the decisions and context from the 2026-10-05 session that aren't in the experiment log.
