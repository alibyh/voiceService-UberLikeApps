"""Arabic → Latin phonetic projection (Hassaniya-tuned).

Goal: collapse "الشارة", "Chara", "Shara", and "Sharah" to (near-)the same string.
Both Arabic-script and Latin-script inputs go through this so the phonetic
retriever operates in one shared space.
"""

from __future__ import annotations

import re

import jellyfish

# Hassaniya pronunciation, not MSA. ج → g is common; ق → g; ث → s; ذ → z.
_AR_TO_LAT = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "s",
    "ج": "g",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "",
    "غ": "gh",
    "ف": "f",
    "ق": "g",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ة": "",  # taa-marbouta is silent in conversational Hassaniya
    "ء": "",
    "ى": "y",
    "ؤ": "w",
    "ئ": "y",
    "أ": "a",
    "إ": "a",
    "آ": "a",
    "ٱ": "a",
    # Persian/Urdu letters that sometimes leak in
    "گ": "g",
    "پ": "p",
    "چ": "ch",
    "ژ": "zh",
}

# Latin digraph normalization (French/English → shared sound).
# Order matters — multi-char patterns first.
_LATIN_DIGRAPHS: list[tuple[str, str]] = [
    ("ch", "sh"),
    ("sch", "sh"),
    ("ph", "f"),
    ("ou", "u"),
    ("oo", "u"),
    ("ee", "i"),
    ("ai", "e"),
    ("ei", "e"),
    ("ay", "e"),
    ("au", "o"),
    ("eau", "o"),
    ("qu", "k"),
    ("ck", "k"),
    ("c", "k"),  # rough but consistent; "ce/ci" becomes "ke/ki" — fine in our space
    ("y", "i"),
    ("j", "j"),
]

_VOWELS = set("aeiou")

# Definite-article noise — strip leading "al-", "el-", "ال" before the rest of
# the word (only when followed by a non-space character).
_ART_RE = re.compile(r"\b(?:al|el)[\s-]?", re.IGNORECASE)


def _ar_to_latin(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in _AR_TO_LAT:
            out.append(_AR_TO_LAT[ch])
        elif ch.isspace():
            out.append(" ")
        elif "؀" <= ch <= "ۿ":
            # Unmapped Arabic char — drop quietly.
            continue
        else:
            out.append(ch.lower())
    return "".join(out)


def _strip_articles(text: str) -> str:
    # Strip Arabic "ال" article when it appears as a leading prefix on a word.
    text = re.sub(r"\bال", "", text)
    # Strip Latin "al-" / "el-".
    text = _ART_RE.sub("", text)
    return text


def _apply_digraphs(text: str) -> str:
    for src, dst in _LATIN_DIGRAPHS:
        text = text.replace(src, dst)
    return text


def _collapse_doubles(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text)


def _strip_short_vowels(text: str) -> str:
    # Phonetic spaces tend to over-weight vowels. Keep them only where they
    # separate consonant clusters; drop standalone vowels at word edges.
    # Simple heuristic: collapse runs of vowels to one.
    return re.sub(r"[aeiou]+", lambda m: m.group(0)[0], text)


def arabic_to_latin_phonetic(text: str) -> str:
    """Project mixed Arabic/Latin text into a single phonetic key."""
    if not text:
        return ""

    # 1. Strip articles first (works in both scripts).
    text = _strip_articles(text)

    # 2. Arabic → Latin char-by-char.
    text = _ar_to_latin(text)

    # 3. Lowercase + strip non-letters except space.
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # 4. Latin digraph normalization.
    text = _apply_digraphs(text)

    # 5. Collapse doubles, normalize vowels, collapse whitespace.
    text = _collapse_doubles(text)
    text = _strip_short_vowels(text)
    # 6. Drop word-final silent 'h' (Latin "Sharah" matches Arabic "شارة").
    text = re.sub(r"([aeiou])h\b", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def phonetic_key(text: str) -> str:
    """One final metaphone squeeze on top of the projection.

    For multi-word strings we encode each token and join, so word order is
    preserved (token-set matching happens at retrieval time, not here).
    """
    projected = arabic_to_latin_phonetic(text)
    if not projected:
        return ""
    parts = [jellyfish.metaphone(tok) or tok for tok in projected.split()]
    return " ".join(p for p in parts if p)
