"""The feature library (v0): one SQLite file that holds a language and everything said in it.

    dictionaries  the frozen symbols + the drawer that turns symbols back into data, versioned and hashed
    speakers      the encoders that write in a dictionary
    items         what was described (one row per image), with its true label, kept for scoring only
    messages      each item's words, with point and time as plain labels (for data over time)
    word_cards    the measured meaning of each symbol: where it appears, what it draws, how stable it is
    scorecards    every measure with its interval, bar and verdict

A dictionary is frozen the moment it is stored: the database refuses any
change to it, so a changed dictionary has to be stored as a new version.
A reader needs only this file: no model file and no experiment script.

v0 limits: set words on images only (fam/set_vqvae.py), one speaker per dictionary.
"""

import hashlib
import io
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import torch

from fam import scorecard, set_vqvae

FROZEN = "a stored dictionary is frozen: store a new version instead"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS dictionaries (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    data_kind TEXT NOT NULL,
    settings TEXT NOT NULL,              -- JSON: recipe, sizes, seed, class names
    weights BLOB NOT NULL,               -- the symbols and the drawer
    weights_sha256 TEXT NOT NULL,
    training_data_sha256 TEXT NOT NULL,
    created TEXT NOT NULL,
    UNIQUE (name, version)
);
CREATE TRIGGER IF NOT EXISTS dictionary_frozen_update BEFORE UPDATE ON dictionaries
BEGIN SELECT RAISE(ABORT, '{FROZEN}'); END;
CREATE TRIGGER IF NOT EXISTS dictionary_frozen_delete BEFORE DELETE ON dictionaries
BEGIN SELECT RAISE(ABORT, '{FROZEN}'); END;

CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries (id),
    name TEXT NOT NULL,
    encoder BLOB NOT NULL,
    encoder_sha256 TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY,
    split TEXT NOT NULL,                 -- 'train' or 'test'
    label INTEGER                        -- true class, for word cards (train) and scoring (test) only
);
CREATE TABLE IF NOT EXISTS messages (
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries (id),
    speaker_id INTEGER NOT NULL REFERENCES speakers (id),
    item_id INTEGER NOT NULL REFERENCES items (item_id),
    words BLOB NOT NULL,                 -- one byte per word
    point TEXT,                          -- which sensor or tower, for data over time (empty for images)
    time TEXT,
    PRIMARY KEY (dictionary_id, item_id)
);
-- The same messages turned around, so "all items containing word 29" is a quick lookup.
CREATE TABLE IF NOT EXISTS message_words (
    dictionary_id INTEGER NOT NULL,
    symbol INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    PRIMARY KEY (dictionary_id, symbol, item_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS word_cards (
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries (id),
    symbol INTEGER NOT NULL,
    uses INTEGER NOT NULL,               -- training items it appears on
    share REAL NOT NULL,                 -- uses as a share of all training items
    class_counts TEXT NOT NULL,          -- JSON: class name -> training items
    top_class TEXT NOT NULL,
    purity REAL NOT NULL,                -- share of its items that are its top class
    appears_with TEXT NOT NULL,          -- JSON: the 5 symbols it shares most items with
    stability REAL,                      -- does it come back in a second training? (empty if not measured)
    picture BLOB NOT NULL,               -- what it draws alone: 28x28 bytes
    PRIMARY KEY (dictionary_id, symbol)
);
CREATE TABLE IF NOT EXISTS scorecards (
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries (id),
    measure TEXT NOT NULL,
    value REAL NOT NULL,
    low REAL,                            -- 95% interval, where there is one
    high REAL,
    bar REAL,
    verdict TEXT NOT NULL,
    note TEXT NOT NULL
);
"""


def _pack(state: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _unpack(blob: bytes) -> dict:
    return torch.load(io.BytesIO(blob))


def weights_hash(state: dict) -> str:
    """SHA-256 of the numbers themselves, in name order, so it doesn't depend on how they were saved."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].numpy().tobytes())
    return digest.hexdigest()


def _split_model(model: set_vqvae.SetVQVAE) -> tuple[dict, dict]:
    """(dictionary part, speaker part): the encoder writes; the symbols and the drawer are the language."""
    state = model.state_dict()
    encoder = {k: v for k, v in state.items() if k.startswith("encoder.")}
    return {k: v for k, v in state.items() if k not in encoder}, encoder


class Library:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._models = {}  # dictionary id -> loaded dictionary, so the weights are unpacked once

    # ---- writing ----

    def add_dictionary(self, name: str, version: str, data_kind: str, model: set_vqvae.SetVQVAE,
                       settings: dict, training_data_sha256: str) -> int:
        """Store a dictionary (symbols + drawer). From now on it can't be changed, only read."""
        state, _ = _split_model(model)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.db:
            row = self.db.execute(
                "INSERT INTO dictionaries (name, version, data_kind, settings, weights, weights_sha256, "
                "training_data_sha256, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, version, data_kind, json.dumps(settings), _pack(state), weights_hash(state),
                 training_data_sha256, created))
        return row.lastrowid

    def add_speaker(self, dictionary_id: int, name: str, model: set_vqvae.SetVQVAE, note: str = "") -> int:
        """Store an encoder that writes in this dictionary."""
        _, encoder = _split_model(model)
        with self.db:
            row = self.db.execute(
                "INSERT INTO speakers (dictionary_id, name, encoder, encoder_sha256, note) VALUES (?, ?, ?, ?, ?)",
                (dictionary_id, name, _pack(encoder), weights_hash(encoder), note))
        return row.lastrowid

    def add_items(self, item_ids: np.ndarray, split: str, labels: np.ndarray) -> None:
        with self.db:
            self.db.executemany("INSERT INTO items (item_id, split, label) VALUES (?, ?, ?)",
                                [(int(i), split, int(y)) for i, y in zip(item_ids, labels)])

    def add_messages(self, dictionary_id: int, speaker_id: int, item_ids: np.ndarray, words: np.ndarray) -> None:
        """Store each item's words. Point and time stay empty here: images have neither."""
        words = words.astype(np.uint8)
        rows, symbols = np.nonzero(scorecard.presence(words))  # each (item, symbol) pair once
        with self.db:
            self.db.executemany(
                "INSERT INTO messages (dictionary_id, speaker_id, item_id, words) VALUES (?, ?, ?, ?)",
                [(dictionary_id, speaker_id, int(i), w.tobytes()) for i, w in zip(item_ids, words)])
            self.db.executemany(
                "INSERT INTO message_words (dictionary_id, symbol, item_id) VALUES (?, ?, ?)",
                [(dictionary_id, int(s), int(item_ids[r])) for r, s in zip(rows, symbols)])

    def add_word_cards(self, dictionary_id: int, cards: list[dict]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT INTO word_cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(dictionary_id, c["symbol"], c["uses"], c["share"], json.dumps(c["class_counts"]), c["top_class"],
                  c["purity"], json.dumps(c["appears_with"]), c["stability"], c["picture"].tobytes())
                 for c in cards])

    def add_score(self, dictionary_id: int, measure: str, value: float, low: float | None = None,
                  high: float | None = None, bar: float | None = None, verdict: str = "", note: str = "") -> None:
        with self.db:
            self.db.execute("INSERT INTO scorecards VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (dictionary_id, measure, value, low, high, bar, verdict, note))

    # ---- reading ----

    def dictionary(self, name: str, version: str) -> dict:
        """A dictionary's id and description (not its weights)."""
        row = self.db.execute(
            "SELECT id, name, version, data_kind, settings, weights_sha256, training_data_sha256, created "
            "FROM dictionaries WHERE name = ? AND version = ?", (name, version)).fetchone()
        return {**dict(row), "settings": json.loads(row["settings"])}

    def verify(self, dictionary_id: int) -> bool:
        """Do the stored weights still match the hash taken when the dictionary was frozen?"""
        row = self.db.execute("SELECT weights, weights_sha256 FROM dictionaries WHERE id = ?",
                              (dictionary_id,)).fetchone()
        return weights_hash(_unpack(row["weights"])) == row["weights_sha256"]

    def load_dictionary(self, dictionary_id: int) -> set_vqvae.SetVQVAE:
        """The symbols and the drawer, ready to read messages. It can't write: the encoder is a speaker's."""
        if dictionary_id not in self._models:
            if not self.verify(dictionary_id):
                raise ValueError("the stored dictionary no longer matches its hash")
            row = self.db.execute("SELECT settings, weights FROM dictionaries WHERE id = ?",
                                  (dictionary_id,)).fetchone()
            model = set_vqvae.SetVQVAE(alphabet=json.loads(row["settings"])["alphabet"])
            model.load_state_dict(_unpack(row["weights"]), strict=False)
            self._models[dictionary_id] = model.eval()
        return self._models[dictionary_id]

    def load_speaker(self, speaker_id: int) -> set_vqvae.SetVQVAE:
        """A full model that can write: the speaker's encoder on top of its dictionary."""
        row = self.db.execute("SELECT dictionary_id, encoder FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
        settings = self.db.execute("SELECT settings FROM dictionaries WHERE id = ?",
                                   (row["dictionary_id"],)).fetchone()["settings"]
        model = set_vqvae.SetVQVAE(alphabet=json.loads(settings)["alphabet"])
        model.load_state_dict({**self.load_dictionary(row["dictionary_id"]).state_dict(), **_unpack(row["encoder"])})
        return model.eval()

    def message(self, dictionary_id: int, item_id: int) -> np.ndarray:
        """One item's words."""
        row = self.db.execute("SELECT words FROM messages WHERE dictionary_id = ? AND item_id = ?",
                              (dictionary_id, item_id)).fetchone()
        return np.frombuffer(row["words"], dtype=np.uint8)

    def messages(self, dictionary_id: int, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every message of a split, in item order: (item ids, words (items, words per item), true labels)."""
        rows = self.db.execute(
            "SELECT m.item_id, m.words, i.label FROM messages m JOIN items i ON i.item_id = m.item_id "
            "WHERE m.dictionary_id = ? AND i.split = ? ORDER BY m.item_id", (dictionary_id, split)).fetchall()
        words = np.frombuffer(b"".join(r["words"] for r in rows), dtype=np.uint8).reshape(len(rows), -1)
        return np.array([r["item_id"] for r in rows]), words, np.array([r["label"] for r in rows])

    def items_with_word(self, dictionary_id: int, symbol: int) -> np.ndarray:
        """All items whose message contains this symbol."""
        rows = self.db.execute("SELECT item_id FROM message_words WHERE dictionary_id = ? AND symbol = ?",
                               (dictionary_id, symbol)).fetchall()
        return np.array([r["item_id"] for r in rows])

    def card(self, dictionary_id: int, symbol: int) -> dict:
        """Everything measured about one symbol."""
        row = self.db.execute("SELECT * FROM word_cards WHERE dictionary_id = ? AND symbol = ?",
                              (dictionary_id, symbol)).fetchone()
        return _card(row)

    def cards(self, dictionary_id: int) -> list[dict]:
        """Every symbol's card, in symbol order."""
        rows = self.db.execute("SELECT * FROM word_cards WHERE dictionary_id = ? ORDER BY symbol", (dictionary_id,))
        return [_card(row) for row in rows]

    def words_meaning(self, dictionary_id: int, class_name: str) -> list[dict]:
        """The symbols that mostly appear on this class, purest first."""
        rows = self.db.execute("SELECT * FROM word_cards WHERE dictionary_id = ? AND top_class = ? "
                               "ORDER BY purity DESC", (dictionary_id, class_name))
        return [_card(row) for row in rows]

    def scores(self, dictionary_id: int) -> list[dict]:
        return [dict(row) for row in self.db.execute("SELECT * FROM scorecards WHERE dictionary_id = ?",
                                                     (dictionary_id,))]

    @torch.no_grad()
    def vectors(self, dictionary_id: int, words: np.ndarray) -> np.ndarray:
        """Look each word up in the dictionary: (items, words) -> (items, words x vector size), for a model to read."""
        table = self.load_dictionary(dictionary_id).vectors()
        return table[torch.tensor(words, dtype=torch.long)].reshape(len(words), -1).numpy()

    @torch.no_grad()
    def draw_words(self, dictionary_id: int, words: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Rebuild images (items, 28, 28) from their words, using only the stored dictionary."""
        model = self.load_dictionary(dictionary_id)
        return np.concatenate([
            model.decode_words(torch.tensor(words[i:i + batch_size], dtype=torch.long)).squeeze(1).numpy()
            for i in range(0, len(words), batch_size)
        ])

    def draw(self, dictionary_id: int, item_id: int) -> np.ndarray:
        """Rebuild one item's image from the library alone."""
        return self.draw_words(dictionary_id, self.message(dictionary_id, item_id)[None])[0]


def _card(row: sqlite3.Row) -> dict:
    card = dict(row)
    card["class_counts"] = json.loads(card["class_counts"])
    card["appears_with"] = json.loads(card["appears_with"])
    card["picture"] = np.frombuffer(card["picture"], dtype=np.uint8).reshape(28, 28)
    return card


@torch.no_grad()
def make_word_cards(model: set_vqvae.SetVQVAE, words: np.ndarray, labels: np.ndarray, classes: list[str],
                    words_second: np.ndarray | None = None) -> list[dict]:
    """Measure every used symbol on labelled TRAINING items: one card (a dict) per symbol.

    words_second: the same items written by a second training of the recipe
    (another seed), to measure whether each symbol comes back (stability).
    """
    vocab = len(model.vectors())
    present = scorecard.presence(words, vocab)
    counts = np.stack([present[labels == c].sum(axis=0) for c in range(len(classes))], axis=1)  # (vocab, classes)
    uses = counts.sum(axis=1)
    together = present.T.astype(np.float32) @ present.astype(np.float32)  # (vocab, vocab) items shared
    np.fill_diagonal(together, 0)
    stable = None if words_second is None else scorecard.symbol_stability(words, words_second, vocab)
    alone = torch.sigmoid(model.drawer(model.vectors())).squeeze(1).numpy()  # each symbol drawn by itself

    cards = []
    for s in np.nonzero(uses)[0]:
        partners = np.argsort(-together[s])[:5]
        cards.append({
            "symbol": int(s),
            "uses": int(uses[s]),
            "share": float(uses[s] / len(words)),
            "class_counts": {classes[c]: int(n) for c, n in enumerate(counts[s])},
            "top_class": classes[counts[s].argmax()],
            "purity": float(counts[s].max() / uses[s]),
            "appears_with": [[int(p), int(together[s, p])] for p in partners],
            "stability": None if stable is None else float(stable[s]),
            "picture": np.round(alone[s] * 255).astype(np.uint8),
        })
    return cards


def card_text(card: dict) -> str:
    """A card as one line of plain text, with the symbol written as a token like <w29>."""
    classes = sorted(card["class_counts"].items(), key=lambda kv: -kv[1])[:3]
    mix = ", ".join(f"{n / card['uses']:.0%} {name}" for name, n in classes if n)
    partners = " ".join(f"<w{p}>" for p, _ in card["appears_with"][:3])
    stable = "" if card["stability"] is None else f" Comes back in a second training {card['stability']:.0%} of the time."
    return (f"<w{card['symbol']}>: on {card['uses']:,} training items ({card['share']:.1%}). "
            f"{mix}. Often with {partners}.{stable}")
