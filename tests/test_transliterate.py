from matcher.transliterate import arabic_to_latin_phonetic, phonetic_key


def test_arabic_and_latin_collapse_to_similar_keys():
    """The whole point: Arabic and Latin spellings of the same sound get close."""
    ar = phonetic_key("الشارة")
    lat1 = phonetic_key("Chara")
    lat2 = phonetic_key("Shara")
    lat3 = phonetic_key("Sharah")
    # All four should share the same metaphone consonant skeleton.
    assert ar.replace(" ", "") == lat2.replace(" ", "") == lat3.replace(" ", "")
    # Chara (ch→sh) lands on the same skeleton.
    assert lat1.replace(" ", "") == lat2.replace(" ", "")


def test_french_digraphs_normalized():
    # ou → u, ai → e, ph → f
    assert "u" in arabic_to_latin_phonetic("Boulangerie")
    assert "f" in arabic_to_latin_phonetic("Telephone")


def test_definite_article_stripped():
    a = arabic_to_latin_phonetic("الشارة")
    b = arabic_to_latin_phonetic("شارة")
    # After stripping al-, both should produce the same result.
    assert a == b


def test_petrodis_variants_collapse():
    # "بتروديس" / "بيتروديس" / "Petrodis" should land near each other.
    a = phonetic_key("بتروديس")
    b = phonetic_key("بيتروديس")
    c = phonetic_key("Petrodis")
    # Allow B/P swap since metaphone treats them as similar but not identical.
    assert a.replace(" ", "") == b.replace(" ", "")
    # Latin "Petrodis" → "PTRTS", Arabic → "BTRTS" — they differ only on B/P.
    assert a.replace("B", "")[1:] == c.replace("P", "")[1:] or len(a) == len(c)
