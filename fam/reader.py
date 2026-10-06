"""The reader (phase 1): an off-the-shelf LLM reads an item's words through their word cards.

    item's 8 words  ->  their word cards as text (fam/library.card_text)  ->  Claude
        ->  {class, confidence 0-1, the words that decided it, a short reason}

No training: the LLM only sees what the library measured about each word, so its
answer can be checked against the true label, and its cited words against the
item's actual message. This is a stand-in; the goal is the developer's own
tailored reader (phase 2), which this sets the baseline for.

The API key is read from ANTHROPIC_API_KEY, or from a .env file in the repo root
(gitignored) with a line ANTHROPIC_API_KEY=...
"""

import json
import os
from pathlib import Path

import anthropic

from fam.library import card_text

MODEL = "claude-opus-5-5"
ROOT = Path(__file__).resolve().parent.parent


def client() -> anthropic.Anthropic:
    """A Claude API client, with the key from the environment or the repo's .env file."""
    if "ANTHROPIC_API_KEY" not in os.environ and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY":
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip('"')
    return anthropic.Anthropic()


def system_prompt(classes: list[str]) -> str:
    return (
        "You read a learned language that describes small greyscale photos of clothing. "
        f"Every photo is one of these classes: {', '.join(classes)}.\n\n"
        "A model turned each photo into 8 words from a dictionary of 256 learned words, written <w0> to <w255>. "
        "Nobody named the words; what each one means was measured afterwards on 60,000 labelled training "
        "photos and written down as its word card: how often it appears, which classes it appears on, "
        "which words it often appears with, and how reliably it comes back when the dictionary is trained again.\n\n"
        "You get one photo's 8 words and their cards. Decide which class the photo is. Base the answer only on "
        "the cards. Give your confidence from 0 to 1 (how likely your answer is right), the 1 to 3 words that "
        "decided it, and one short sentence on why."
    )


def answer_schema(classes: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": classes},
            "confidence": {"type": "number"},
            "key_words": {"type": "array", "items": {"type": "integer"}},
            "reason": {"type": "string"},
        },
        "required": ["class", "confidence", "key_words", "reason"],
        "additionalProperties": False,
    }


def item_text(words, cards_by_symbol: dict) -> str:
    """The user message: the item's words, then each distinct word's card."""
    listed = " ".join(f"<w{int(w)}>" for w in words)
    lines = [card_text(cards_by_symbol[int(w)]) for w in dict.fromkeys(int(w) for w in words)]
    return f"Words: {listed}\n\nWord cards:\n" + "\n".join(lines)


def read(api: anthropic.Anthropic, words, cards_by_symbol: dict, classes: list[str]) -> dict:
    """Ask Claude to read one item. Returns its answer plus the token usage (or a refusal)."""
    response = api.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, the API retries on another model
        system=system_prompt(classes),
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": answer_schema(classes)}},
        messages=[{"role": "user", "content": item_text(words, cards_by_symbol)}],
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
             "model": response.model}
    if response.stop_reason == "refusal":
        return {"refused": True, **usage}
    text = next(b.text for b in response.content if b.type == "text")
    return {"refused": False, **json.loads(text), **usage}
