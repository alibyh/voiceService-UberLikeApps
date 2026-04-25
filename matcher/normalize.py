"""Stage 2: text normalization.

Symmetry is the contract: every transformation here must be applied identically
to the query and to every variant at index-build time. If you change this file,
rebuild the indexes.
"""

from __future__ import annotations

import re
import unicodedata

import pyarabic.araby as araby

# Tokens that add no discriminative value but appear in many place names.
# We never *delete* them; we emit two forms (with and without) so a query
# missing the prefix still matches, and so does a query that includes it.
_STOP_PREFIXES = (
    "محطة",
    "ستاصيون",
    "ستاسيون",
    "station",
    "gas station",
    "route",
    "طريق",
    "boulangerie",
    "boulangeerie",
    "bolanjri",
    "بولانجري",
    "بولانجيري",
    "بولانشري",
    "marche",
    "marché",
    "marchée",
    "مرصة",
    "مرشي",
    "مرشيه",
    "مارشي",
    "souk",
    "سوق",
    "carrefour",
    "كارفور",
    "كرفور",
    "mosque",
    "mosquee",
    "mosquée",
    "مسجد",
    "مسيد",
    "موسكي",
    "جامع",
)

_AR_PUNCT = "،؟؛«»ـ"
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _normalize_arabic(text: str) -> str:
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    # alef variants → bare alef
    text = re.sub(r"[إأآٱ]", "ا", text)
    # ى → ي, ة → ه, ؤ → و, ئ → ي
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.translate(_ARABIC_DIGITS)
    return text


def _normalize_latin(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    # drop combining accents (é → e)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def _strip_punct(text: str) -> str:
    # Drop common punctuation but keep word boundaries via space.
    text = re.sub(rf"[{re.escape(_AR_PUNCT)}]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", " ", text)
    return text


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """Single canonical normalized form."""
    if not text:
        return ""
    text = _normalize_arabic(text)
    text = _normalize_latin(text)
    text = _strip_punct(text)
    return _collapse_ws(text)


def _strip_stop_prefixes(text: str) -> str:
    """Remove leading and trailing stop-prefix tokens (post-normalization match)."""
    tokens = text.split()
    if not tokens:
        return text

    changed = True
    while changed and tokens:
        changed = False
        # leading
        for sp_tokens in _NORMALIZED_STOP_TOKENS:
            n = len(sp_tokens)
            if len(tokens) > n and [t.lower() for t in tokens[:n]] == sp_tokens:
                tokens = tokens[n:]
                changed = True
                break
        # trailing — also drop, e.g. "petrodis station"
        for sp_tokens in _NORMALIZED_STOP_TOKENS:
            n = len(sp_tokens)
            if len(tokens) > n and [t.lower() for t in tokens[-n:]] == sp_tokens:
                tokens = tokens[:-n]
                changed = True
                break

    return " ".join(tokens)


# Pre-normalize the stop-prefix list so comparisons happen in the same space
# as the input text. Built lazily after `normalize` is defined above.
_NORMALIZED_STOP_TOKENS: list[list[str]] = sorted(
    {tuple(normalize(p).split()) for p in _STOP_PREFIXES if normalize(p)},
    key=lambda toks: -len(toks),  # longer phrases first to avoid shadowing
)
_NORMALIZED_STOP_TOKENS = [list(t) for t in _NORMALIZED_STOP_TOKENS]


def normalize_forms(text: str) -> list[str]:
    """Return all normalized forms to index/query.

    Always returns at least one element. If stripping stop-prefixes produces a
    different non-empty string, it is included as a second form.
    """
    canonical = normalize(text)
    if not canonical:
        return [""]

    stripped = _strip_stop_prefixes(canonical)
    if stripped and stripped != canonical:
        return [canonical, stripped]
    return [canonical]
