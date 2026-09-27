import pytest

from ductmark.labels import DuctSize, format_ft_in, has_size_mark, infer_scale, parse_scale, pick, scale_notes, size_readings


@pytest.mark.parametrize(
    "text, expected",
    [
        ('12"ø', [DuctSize(12)]),
        ('12"Φ', [DuctSize(12)]),
        ('18"g', [DuctSize(18)]),
        ('12"0', [DuctSize(12)]),
        ("80", [DuctSize(8)]),
        ("|140]", [DuctSize(14)]),
        ("8110", [DuctSize(8)]),
        ('22"X14*', [DuctSize(22, 14)]),
        ("22x14", [DuctSize(22, 14)]),
    ],
)
def test_size_readings(text, expected):
    assert size_readings(text) == expected


@pytest.mark.parametrize("text", ['3/4" U/C', '+48"AFF', "BDD", "RTU-1", "1/4\"=1'-0\"", "10'"])
def test_non_size_text_has_no_readings(text):
    assert size_readings(text) == []


def test_size_mark_separates_labels_from_tags():
    assert has_size_mark('4"ø') and has_size_mark('12"0') and has_size_mark("22x14")
    assert not has_size_mark("400") and not has_size_mark('6"')


def test_pick_uses_measured_width():
    assert pick(size_readings("300"), 12.0) is None
    assert pick(size_readings("120"), 12.1) == DuctSize(12)
    assert pick([DuctSize(22, 14)], 14.2) == DuctSize(22, 14)


@pytest.mark.parametrize(
    "text, ppi",
    [("1/4\"=1'-0\"", 1.5), ("1/8\"=1'-0\"", 0.75), ("3/16\"=1'-0\"", 1.125), ("1\"=10'", 0.6), ("1:50", 1.44)],
)
def test_parse_scale(text, ppi):
    assert parse_scale(text) == pytest.approx(ppi)


def test_infer_scale_from_labels():
    obs = [([DuctSize(8)], 12.0), ([DuctSize(14)], 21.1), ([DuctSize(18)], 27.0), (size_readings("400"), 30.0)]
    assert infer_scale(obs) == ("1/4\"=1'-0\"", 1.5)


def test_infer_scale_needs_agreement():
    with pytest.raises(ValueError, match="too few"):
        infer_scale([([DuctSize(8)], 12.0)])


QUARTER, EIGHTH = "1/4\"=1'-0\"", "1/8\"=1'-0\""
# 8" and 12" read at 1/4"=1'-0", or 16" and 24" read at 1/8"=1'-0": two votes each.
TIED = [([DuctSize(8)], 12.0), ([DuctSize(12)], 18.0), ([DuctSize(16)], 12.0), ([DuctSize(24)], 18.0)]


def test_tied_scale_votes_are_rejected():
    with pytest.raises(ValueError, match="fit both"):
        infer_scale(TIED)


def test_narrow_scale_lead_is_rejected():
    with pytest.raises(ValueError, match="fit both"):
        infer_scale(TIED + [([DuctSize(10)], 15.0)])  # 3 votes to 2


def test_scale_noted_on_the_sheet_settles_a_tie():
    assert infer_scale(TIED, notes=[QUARTER]) == (QUARTER, 1.5)
    assert infer_scale(TIED, notes=["1/8\"=1'"]) == (EIGHTH, 0.75)


def test_scale_noted_on_the_sheet_must_agree_with_labels():
    obs = [([DuctSize(8)], 12.0), ([DuctSize(14)], 21.1), ([DuctSize(18)], 27.0)]
    with pytest.raises(ValueError, match="sheet notes"):
        infer_scale(obs, notes=[EIGHTH])


def test_scale_notes_are_found_in_sheet_text():
    text = "MECHANICAL FLOOR PLAN  SCALE 1/4\" = 1'-0\"  9/23/2025 12:02pm  1\"=20'  1/4°=1"
    assert scale_notes([text]) == [QUARTER, "1\"=20'"]


def test_format_ft_in():
    assert format_ft_in(11.333) == "11'-4\""
    assert format_ft_in(11.99) == "12'-0\""
