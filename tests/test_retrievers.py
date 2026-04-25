"""Smoke tests that the retrievers can build and rank a known place."""

import json
from pathlib import Path

import pytest

from matcher.retrievers.lexical import LexicalIndex
from matcher.retrievers.phonetic import PhoneticIndex


@pytest.fixture(scope="module")
def places():
    path = Path(__file__).resolve().parent.parent / "matcher" / "data" / "places.json"
    return [p for p in json.loads(path.read_text(encoding="utf-8")) if "id" in p]


@pytest.fixture(scope="module")
def lexical(places):
    return LexicalIndex.build(places)


@pytest.fixture(scope="module")
def phonetic(places):
    return PhoneticIndex.build(places)


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("بتروديس الشارة", 1844),
        ("Petrodis Chara", 1844),
        ("petro dis route shara", 1844),
        ("ولد سبرو", 13),
        ("كابيتال", 10),
        ("عين الطلح", 11),
    ],
)
def test_expected_id_in_lexical_top_15(lexical, query, expected_id):
    ids = [c.place_id for c in lexical.search(query, top_k=15)]
    assert expected_id in ids, f"{expected_id} not in top-15 lexical for {query!r}: {ids}"


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("بتروديس الشارة", 1844),
        ("Petrodis Chara", 1844),
        ("ولد سبرو", 13),
        ("كابيتال", 10),
        ("عين الطلح", 11),
    ],
)
def test_expected_id_in_phonetic_top_15(phonetic, query, expected_id):
    ids = [c.place_id for c in phonetic.search(query, top_k=15)]
    assert expected_id in ids, f"{expected_id} not in top-15 phonetic for {query!r}: {ids}"
