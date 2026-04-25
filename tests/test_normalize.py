from matcher.normalize import normalize, normalize_forms


def test_strips_diacritics_and_alef_variants():
    assert normalize("بَيْتُ الْأَمَلْ") == "بيت الامل"
    assert normalize("إسلام") == "اسلام"
    assert normalize("آمنة") == "امنه"


def test_unifies_yaa_and_taa_marbouta():
    assert normalize("الشارة") == "الشاره"
    assert normalize("ذكرى") == "ذكري"


def test_latin_lowercases_and_strips_accents():
    assert normalize("Marché Capitale") == "marche capitale"
    assert normalize("PETRODIS") == "petrodis"


def test_mixed_script_preserved():
    out = normalize("Petrodis طريق الشارة")
    assert "petrodis" in out
    assert "طريق" in out
    assert "الشاره" in out


def test_normalize_forms_emits_stripped_when_stop_prefix_present():
    forms = normalize_forms("محطة بتروديس")
    assert "محطه بتروديس" in forms
    assert "بتروديس" in forms


def test_normalize_forms_single_when_no_stop_prefix():
    forms = normalize_forms("بتروديس الشارة")
    assert forms == ["بتروديس الشاره"]


def test_symmetry_query_and_variant_match():
    """Both sides of the index/query split must produce the same normalized form."""
    variant = normalize("بيترو ديس طريق الشارة")
    query = normalize("بِيتْرُو دِيس طريق الشّارَة")
    assert variant == query
