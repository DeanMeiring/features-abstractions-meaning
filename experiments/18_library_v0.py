"""Experiment 18: Library v0. Can the words be stored once and reused from the library alone?

Until now model A's words lived in a model file that model B opened directly.
Here experiment 10's set words go into one SQLite file (fam/library.py):

    dictionary  the 256 symbols + the drawer, frozen, versioned and hashed
    speaker     experiment 10's encoder, which writes in that dictionary
    messages    8 words for each of the 70,000 images
    word cards  the measured meaning of each symbol, from TRAINING labels only
    scorecard   every measure with its interval, bar and verdict

Then the library is closed and opened again, and everything below reads it
only through its query functions: no model file. No new encoder is trained.

Success bar, fixed in the README before running:
    round trip:     the test words read back are exactly the model's, and the test images
                    rebuilt from the library alone give the same rebuild error (6 decimals)
    reuse:          model B reading through the library gets exactly the same accuracy on
                    each of the same 20 picks of 50 labels as reading the model directly
    frozen:         changing or deleting a stored dictionary fails, and its hash verifies
    cards carry it: each test image's class predicted from its word cards alone (its 8 words
                    vote with their cards' class shares) reaches >= 70%, whole 95% interval
Needs experiments 10 and 14 first (14 for the second training that stability compares with).
"""

import hashlib
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fam.compute  # noqa: E402,F401  (first: caps CPU/GPU use at 30%)

import numpy as np
import torch

from fam import library, scorecard, set_vqvae
from fam.images import CLASSES, RAW, load_fashion_mnist

MODELS = ROOT / "data" / "models"
LIBRARY = ROOT / "data" / "library_v0.db"
NAME, VERSION = "fashion-set8", "v1"
start_all = time.time()

train_images, train_labels = load_fashion_mnist("train")
test_images, test_labels = load_fashion_mnist("t10k")
train_ids, test_ids = np.arange(60000), np.arange(60000, 70000)


def load(name):
    model = set_vqvae.SetVQVAE()
    model.load_state_dict(torch.load(MODELS / name))
    return model.eval()


# Step 1: the writer's side. Experiment 10's model writes every image as 8 words.
model = load("fashion_set8.pt")
train_w, test_w = set_vqvae.words_of(model, train_images), set_vqvae.words_of(model, test_images)
second_w = set_vqvae.words_of(load("fashion_set8_seed7.pt"), train_images)  # same recipe, another seed

# Step 2: build the library (from scratch each run: it is derived from the models, like a picture).
if LIBRARY.exists():
    LIBRARY.unlink()
t = time.time()
lib = library.Library(LIBRARY)
settings = {"recipe": "set_vqvae", "alphabet": False, "symbols": 256, "vector_size": 16, "words_per_message": 8,
            "seed": 42, "epochs": 6, "classes": CLASSES}
data_hash = hashlib.sha256((RAW / "train-images-idx3-ubyte.gz").read_bytes()).hexdigest()
d = lib.add_dictionary(NAME, VERSION, "28x28 greyscale clothing images", model, settings, data_hash)
speaker = lib.add_speaker(d, "original", model, note="experiment 10's encoder, trained with the dictionary")
lib.add_items(train_ids, "train", train_labels)
lib.add_items(test_ids, "test", test_labels)
lib.add_messages(d, speaker, train_ids, train_w)
lib.add_messages(d, speaker, test_ids, test_w)
lib.add_word_cards(d, library.make_word_cards(model, train_w, train_labels, CLASSES, words_second=second_w))
lib.db.close()
build_s = time.time() - t
print(f"Built {LIBRARY.name}: {LIBRARY.stat().st_size / 1e6:.1f} MB in {build_s:.0f} s "
      f"(70,000 messages x 8 bytes = {70000 * 8 / 1e6:.2f} MB of words)")

# Step 3: the reader's side. Open the file again; from here on, only library queries.
lib = library.Library(LIBRARY)
d = lib.dictionary(NAME, VERSION)["id"]
_, lib_train_w, lib_train_labels = lib.messages(d, "train")
_, lib_test_w, lib_test_labels = lib.messages(d, "test")
cards = lib.cards(d)
print(f"Read back: {len(lib_train_w):,} + {len(lib_test_w):,} messages, {len(cards)} word cards\n")
print("Three example cards:")
for card in sorted(cards, key=lambda c: -c["uses"])[:3]:
    print("  " + library.card_text(card))

# Check 1: round trip.
pixels = test_images / 255.0
with torch.no_grad():
    direct = np.concatenate([model.decode_words(torch.tensor(test_w[i:i + 1000], dtype=torch.long)).squeeze(1).numpy()
                             for i in range(0, len(test_w), 1000)])
error_model = float(((direct - pixels) ** 2).mean())
error_library = float(((lib.draw_words(d, lib_test_w) - pixels) ** 2).mean())
same_words = np.array_equal(lib_test_w, test_w) and np.array_equal(lib_train_w, train_w)
round_trip = same_words and round(error_model, 6) == round(error_library, 6)

# Check 2: model B reads through the library, on the same 20 picks of 50 labels as experiments 6-16.
rng = np.random.default_rng(42)
picks = [rng.choice(len(train_images), size=50, replace=False) for _ in range(20)]


def model_vectors(words):
    with torch.no_grad():
        return model.codebook[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()


scores_model = scorecard.few_label_scores(model_vectors(train_w), train_labels, model_vectors(test_w), test_labels, picks)
scores_library = scorecard.few_label_scores(lib.vectors(d, lib_train_w), lib_train_labels,
                                            lib.vectors(d, lib_test_w), lib_test_labels, picks)
reuse = np.array_equal(scores_model, scores_library)

# Check 3: frozen. Both attempts must be refused by the database.
refused = 0
for attempt in ("UPDATE dictionaries SET version = 'v2' WHERE id = ?", "DELETE FROM dictionaries WHERE id = ?"):
    try:
        with lib.db:
            lib.db.execute(attempt, (d,))
    except sqlite3.DatabaseError:
        refused += 1
frozen = refused == 2 and lib.verify(d) and lib.dictionary(NAME, VERSION)["id"] == d

# Check 4: cards carry the meaning. Each word votes with its card's class shares; no model is trained.
shares = np.zeros((256, len(CLASSES)))
for card in cards:
    shares[card["symbol"]] = [card["class_counts"][c] / card["uses"] for c in CLASSES]
voted = shares[lib_test_w].sum(axis=1).argmax(axis=1)
vote, vote_low, vote_high = scorecard.interval((voted == lib_test_labels).astype(float))
vote_verdict = scorecard.verdict(vote_low, vote_high, 0.70)

# Reported, no bar: how long a reader waits for each kind of query.
trousers = lib.words_meaning(d, "Trouser")
queries = {
    "card(symbol)": lambda: lib.card(d, cards[0]["symbol"]),
    "words_meaning('Trouser')": lambda: lib.words_meaning(d, "Trouser"),
    "items_with_word(symbol)": lambda: lib.items_with_word(d, trousers[0]["symbol"]),
    "message(item)": lambda: lib.message(d, 60000),
    "draw(item)": lambda: lib.draw(d, 60000),
}
timings = {}
for name, query in queries.items():
    t = time.time()
    for _ in range(20):
        query()
    timings[name] = (time.time() - t) / 20 * 1000

# The scorecard goes into the library too.
few, few_low, few_high = scorecard.interval(scores_library)
lib.add_score(d, "purity", scorecard.purity(lib_train_w, lib_train_labels), note="training items")
lib.add_score(d, "stability between two trainings", scorecard.stability(train_w, second_w), bar=0.70,
              verdict="FAIL", note="experiment 14's bar; seeds 42 and 7")
lib.add_score(d, "rebuild error", error_library, note="test images, from the library alone")
lib.add_score(d, "few-label accuracy, 50 labels", few, few_low, few_high, note="linear model B, 20 picks, via the library")
lib.add_score(d, "class from word cards alone", vote, vote_low, vote_high, bar=0.70, verdict=vote_verdict,
              note="test images; each word votes with its card's class shares")

print(f"\nWords meaning 'Trouser': {len(trousers)} symbols, purest "
      + ", ".join(f"<w{c['symbol']}> {c['purity']:.0%}" for c in trousers[:4]))
print(f"Items containing <w{trousers[0]['symbol']}>: {len(lib.items_with_word(d, trousers[0]['symbol'])):,}")
print("\nTime per query (ms): " + " | ".join(f"{name} {ms:.1f}" for name, ms in timings.items()))

print("\nVerdict:")
print(f"  Round trip: words identical {same_words}; rebuild error library {error_library:.6f} vs model {error_model:.6f}  "
      f"{'PASS' if round_trip else 'FAIL'}")
print(f"  Reuse through the library: {scores_library.mean():.1%} vs model directly {scores_model.mean():.1%}, "
      f"identical on all 20 picks: {reuse}  {'PASS' if reuse else 'FAIL'}")
print(f"  Frozen: {refused}/2 changes refused, hash verifies {lib.verify(d)}  {'PASS' if frozen else 'FAIL'}")
print(f"  Cards carry the meaning: {vote:.1%} [{vote_low:.1%}, {vote_high:.1%}] (bar: >= 70%, whole interval)  {vote_verdict}")
print(f"\nTotal time: {(time.time() - start_all) / 60:.1f} min")
