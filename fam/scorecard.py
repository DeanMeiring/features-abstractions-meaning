"""The word scorecard: the same tests for every word recipe, so recipes can be compared.

    meaning   purity (does a word point to one class?) and few-label accuracy
    compact   bytes per item and rebuild error (measured by the caller)
    units     bag of words (counts only) vs reading the words in order
    stable    train twice with different seeds: do the same words come back?

Words are given as (items, words per item) arrays of symbol numbers 0-255.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def presence(words: np.ndarray, vocab: int = 256) -> np.ndarray:
    """(items, k) symbol numbers -> (items, vocab) True where the symbol appears in the item."""
    out = np.zeros((len(words), vocab), dtype=bool)
    out[np.repeat(np.arange(len(words)), words.shape[1]), words.reshape(-1)] = True
    return out


def purity(words: np.ndarray, labels: np.ndarray, vocab: int = 256) -> float:
    """Usage-weighted share of each symbol's items that belong to its most common class.

    1 / n_classes = the symbol says nothing about the class; 100% = one class only.
    """
    present = presence(words, vocab).astype(np.float32)
    hits = np.stack([present[labels == c].sum(axis=0) for c in np.unique(labels)], axis=1)  # (vocab, classes)
    used = hits.sum(axis=1) > 0
    return float(hits[used].max(axis=1).sum() / hits[used].sum())


def stability(words_a: np.ndarray, words_b: np.ndarray, vocab: int = 256) -> float:
    """Do two trainings (different seeds) write the same items with matching symbols?

    Each symbol of training A is matched to the training-B symbol it most often
    shares an item with. Stability = how often that partner is present wherever
    the symbol is, averaged over both directions (A->B and B->A). 100% = the two
    trainings are the same language with renamed symbols.
    """
    def one_way(a, b):
        pa, pb = presence(a, vocab), presence(b, vocab)
        together = pa.T.astype(np.float32) @ pb.astype(np.float32)  # (vocab A, vocab B) items shared
        partner = together.argmax(axis=1)
        return float((pa & pb[:, partner]).sum() / pa.sum())

    return (one_way(words_a, words_b) + one_way(words_b, words_a)) / 2


def stability_by_chance(words_a: np.ndarray, words_b: np.ndarray, vocab: int = 256, seed: int = 0) -> float:
    """Stability if training B's items were shuffled, i.e. no real match: the floor to compare against.

    Common symbols match by luck (a symbol on most items is "present" anywhere),
    so a stability score only means something above this.
    """
    shuffled = words_b[np.random.default_rng(seed).permutation(len(words_b))]
    return stability(words_a, shuffled, vocab)


def same_symbols(words_a: np.ndarray, words_b: np.ndarray, vocab: int = 256) -> float:
    """Two speakers of the SAME dictionary: share of symbols they both wrote for the same item.

    Symbols are a set (order doesn't matter), so this counts the overlap of the
    two sets, as a share of the symbols per item. 100% = identical messages.
    """
    overlap = np.minimum(bag_of_words(words_a, vocab), bag_of_words(words_b, vocab)).sum(axis=1)
    return float((overlap / words_a.shape[1]).mean())


def bag_of_words(words: np.ndarray, vocab: int = 256) -> np.ndarray:
    """How often each symbol appears in the item, wherever it is."""
    counts = np.zeros((len(words), vocab), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(len(words)), words.shape[1]), words.reshape(-1)), 1)
    return counts


def few_label_accuracy(train_x, train_y, test_x, test_y, picks) -> float:
    """Mean test accuracy of a linear model B trained on each pick of labelled items."""
    return float(np.mean([
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        .fit(train_x[p], train_y[p]).score(test_x, test_y)
        for p in picks
    ]))
