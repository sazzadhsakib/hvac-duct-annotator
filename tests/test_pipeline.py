import csv

import pymupdf
import pytest

from ductmark.annotate import annotate, save
from ductmark.geometry import Run, neighbors
from ductmark.labels import DuctSize
from ductmark.ocr import TextBox
from ductmark.pipeline import _assign_labels, _grow, detect

PPI = 1.5  # 1/4"=1'-0"


def run(a, b, width, weight=1.44, walls=()):
    return Run(a, b, width, weight, frozenset(walls))


def test_label_goes_to_the_duct_it_sits_in():
    runs = [run((0, 0), (300, 0), 18, walls=(0, 1)), run((0, 40), (300, 40), 12, walls=(2, 3))]
    labelled = _assign_labels(runs, [TextBox('12"ø', (150, 0)), TextBox('8"ø', (150, 40))], PPI)
    assert labelled == {0: (DuctSize(12), '12"ø'), 1: (DuctSize(8), '8"ø')}


def test_rectangular_label_matches_its_plan_width():
    runs = [run((0, 0), (300, 0), 33, walls=(0, 1))]
    assert _assign_labels(runs, [TextBox('22"x14"', (150, 0))], PPI) == {0: (DuctSize(22, 14), '22"x14"')}


def test_only_marked_labels_attach_from_beside_a_duct():
    runs = [run((0, 0), (300, 0), 6)]
    assert _assign_labels(runs, [TextBox('4"ø', (150, 10))], PPI) == {0: (DuctSize(4), '4"ø')}
    assert _assign_labels(runs, [TextBox("40", (150, 10))], PPI) == {}


def test_label_midway_between_two_matching_ducts_is_not_guessed():
    runs = [run((0, 0), (300, 0), 18, walls=(0, 1)), run((0, 40), (300, 40), 18, walls=(2, 3))]
    assert _assign_labels(runs, [TextBox('12"ø', (150, 20))], PPI) == {}


def test_label_past_the_end_of_a_run_is_not_given_to_it():
    runs = [run((0, 0), (100, 0), 18)]
    assert _assign_labels(runs, [TextBox('12"ø', (120, 0))], PPI) == {}


def test_run_with_conflicting_labels_takes_the_majority_or_none():
    runs = [run((0, 0), (300, 0), 18)]
    tie = [TextBox('12"ø', (50, 0)), TextBox('13"ø', (150, 0))]
    assert _assign_labels(runs, tie, PPI) == {}
    assert _assign_labels(runs, tie + [TextBox('12"ø', (250, 0))], PPI) == {0: (DuctSize(12), '12"ø')}


def test_growth_inherits_same_width_and_filters_the_rest():
    runs = [
        run((0, 0), (300, 0), 18, walls=(0, 1)),  # labelled
        run((318, 18), (318, 200), 18, walls=(2, 3)),  # elbow, same width: inherits
        run((150, 9), (150, 120), 30, walls=(4, 5)),  # tee, other width, same lineweight: measured
        run((300, -9), (300, -150), 30, weight=0.18, walls=(6, 7)),  # different lineweight: dropped
    ]
    accepted = _grow(runs, neighbors(runs), {0: (DuctSize(12), '12"ø')}, PPI)
    assert accepted[1] == (DuctSize(12), "inferred", "")
    assert accepted[2] == (None, "measured", "")
    assert 3 not in accepted


def test_gap_between_adjacent_ducts_is_not_grown_into():
    runs = [
        run((0, 0), (300, 0), 18, walls=(0, 1)),
        run((0, 30), (300, 30), 18, walls=(2, 3)),
        run((0, 15), (300, 15), 12, walls=(1, 2)),  # space between the two ducts
    ]
    labels = {0: (DuctSize(12), '12"ø'), 1: (DuctSize(12), '12"ø')}
    assert 2 not in _grow(runs, neighbors(runs), labels, PPI)


@pytest.fixture
def drawing():
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=400)
    for y, width, label in ((100, 18, '12"ø'), (250, 12, '8"ø')):
        page.draw_line((100, y), (600, y), width=1.44)
        page.draw_line((100, y + width), (600, y + width), width=1.44)
        page.insert_text((330, y + width / 2 + 3), label, fontsize=8)
    yield doc, page
    doc.close()


def test_text_layer_drawing_end_to_end(drawing, tmp_path):
    doc, page = drawing
    takeoff = detect(page)
    assert takeoff.scale == "1/4\"=1'-0\""
    assert [(str(d.size), round(d.length_ft, 1)) for d in takeoff.ducts] == [('12"ø', 27.8), ('8"ø', 27.8)]

    annotate(page, takeoff)
    pdf, png, report = save(doc, page, takeoff, tmp_path, "plan")
    assert pdf.exists() and png.exists()
    with report.open() as f:
        rows = list(csv.DictReader(f))
    assert [(r["id"], r["size"], r["length"], r["source"]) for r in rows] == [
        ("D1", '12"ø', "27'-9\"", "label"),
        ("D2", '8"ø', "27'-9\"", "label"),
    ]


def test_explicit_scale_overrides_a_disagreeing_sheet_note(drawing):
    _, page = drawing
    page.insert_text((620, 380), "SCALE: 1/8\" = 1'-0\"", fontsize=8)
    with pytest.raises(ValueError, match="sheet notes"):
        detect(page)
    takeoff = detect(page, scale="1/4\"=1'-0\"")
    assert [str(d.size) for d in takeoff.ducts] == ['12"ø', '8"ø']


def test_scale_that_no_label_agrees_with_is_rejected(drawing):
    _, page = drawing
    with pytest.raises(ValueError, match="no duct size label"):
        detect(page, scale="1/8\"=1'-0\"")


@pytest.fixture
def callouts():
    doc = pymupdf.open()
    page = doc.new_page(width=800, height=500)

    def walls(a, b, c, d):
        page.draw_line(a, b, width=1.44)
        page.draw_line(c, d, width=1.44)

    walls((100, 100), (600, 100), (100, 118), (600, 118))  # 12" duct
    walls((300, 118), (300, 220), (315, 118), (315, 220))  # 10" branch off it, bare "10" label
    walls((100, 250), (600, 250), (100, 262), (600, 262))  # 8" duct with a "7" callout
    walls((100, 350), (400, 350), (100, 372.5), (400, 372.5))  # 15" table cell holding "150"
    for point, text in (((560, 259), "7"), ((330, 112), '12"ø'), ((303, 172), "10"), ((330, 259), '8"ø'), ((240, 364), "150")):
        page.insert_text(point, text, fontsize=8)
    yield page
    doc.close()


def test_bare_numbers_need_confirmed_ductwork(callouts):
    ducts = {round(d.run.midpoint[1]): (str(d.size), d.source) for d in detect(callouts).ducts}
    assert ducts == {
        109: ('12"ø', "label"),
        169: ('10"ø', "label"),  # bare number on a branch of labelled ductwork
        256: ('8"ø', "label"),  # the explicit label beats the "7" callout
    }  # the "150" table cell is not a duct


def scaled_drawing(ppi, inches):
    """A labelled duct of the given size plus a 12" one, drawn at the given scale."""
    doc = pymupdf.open()
    page = doc.new_page(width=1400, height=700)
    for y, size in ((100, inches), (450, 12)):
        width = size * ppi
        page.draw_line((200, y), (1100, y), width=1.44)
        page.draw_line((200, y + width), (1100, y + width), width=1.44)
        page.insert_text((600, y + width / 2 + 3), f'{size}"ø', fontsize=8)
    return doc, page


@pytest.mark.parametrize("scale, ppi, inches", [("1/2\"=1'-0\"", 3.0, 36), ("1/8\"=1'-0\"", 0.75, 4)])
def test_given_scale_sets_the_duct_width_range(scale, ppi, inches):
    # 36" at 1/2" is 108 pt and 4" at 1/8" is 3 pt: both outside the 4-80 pt paper defaults.
    doc, page = scaled_drawing(ppi, inches)
    assert sorted(str(d.size) for d in detect(page, scale).ducts) == sorted([f'{inches}"ø', '12"ø'])
    doc.close()


def test_riser_box_size_follows_the_scale():
    doc, page = scaled_drawing(3.0, 36)
    page.draw_rect(pymupdf.Rect(92, 100, 200, 208), width=1.44)  # 36" riser box at the duct's end
    page.draw_line((92, 100), (200, 208), width=1.44)
    page.draw_line((92, 208), (200, 100), width=1.44)
    systems = {str(d.size): d.system for d in detect(page, "1/2\"=1'-0\"").ducts}
    assert systems['36"ø'] == "supply"
    doc.close()
