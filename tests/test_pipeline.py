import csv

import pymupdf
import pytest

from ductmark.annotate import annotate, save
from ductmark.geometry import Run
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


def test_only_marked_labels_attach_from_beside_a_duct():
    runs = [run((0, 0), (300, 0), 6)]
    assert _assign_labels(runs, [TextBox('4"ø', (150, 10))], PPI) == {0: (DuctSize(4), '4"ø')}
    assert _assign_labels(runs, [TextBox("40", (150, 10))], PPI) == {}


def test_growth_inherits_same_width_and_filters_the_rest():
    runs = [
        run((0, 0), (300, 0), 18, walls=(0, 1)),  # labelled
        run((318, 18), (318, 200), 18, walls=(2, 3)),  # elbow, same width: inherits
        run((150, 9), (150, 120), 30, walls=(4, 5)),  # tee, other width, same lineweight: measured
        run((300, -9), (300, -150), 30, weight=0.18, walls=(6, 7)),  # different lineweight: dropped
    ]
    accepted = _grow(runs, {0: (DuctSize(12), '12"ø')}, PPI)
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
    assert 2 not in _grow(runs, labels, PPI)


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


def test_scale_that_no_label_agrees_with_is_rejected(drawing):
    _, page = drawing
    with pytest.raises(ValueError, match="no duct size label"):
        detect(page, scale="1/8\"=1'-0\"")
