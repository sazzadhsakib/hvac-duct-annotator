from collections import Counter
from pathlib import Path

import pytest

from ductmark.geometry import merge_collinear, pair_walls
from ductmark.labels import DuctSize
from ductmark.pipeline import detect
from ductmark.vector import dark_segments, load_page

SAMPLE = Path(__file__).parents[1] / "samples" / "testset2.pdf"


@pytest.fixture(scope="module")
def page():
    doc, page = load_page(str(SAMPLE))
    yield page
    doc.close()


def find(runs, x, y, tol=5.0):
    return next(r for r in runs if abs(r.midpoint[0] - x) < tol and abs(r.midpoint[1] - y) < tol)


@pytest.mark.parametrize(
    "x, y, width, length",
    [
        (782, 456, 27.0, 338),  # 18"ø grease duct
        (1485, 526, 21.1, 450),  # 14"ø dining supply
        (1488, 756, 21.1, 443),
        (555, 528, 11.9, 204),  # 8"ø branches
        (534, 672, 11.9, 174),
        (881, 712, 18.1, 143),  # 12"ø diagonal
        (1125, 613, 33.1, 89),  # 22"x14" diagonal
    ],
)
def test_reference_runs_are_found(page, x, y, width, length):
    runs = merge_collinear(pair_walls(dark_segments(page)))
    run = find(runs, x, y)
    assert run.width == pytest.approx(width, abs=0.5)
    assert run.length >= length - 1


@pytest.mark.slow
def test_full_takeoff(page):
    takeoff = detect(page)
    assert takeoff.scale == "1/4\"=1'-0\""
    assert Counter(d.source for d in takeoff.ducts) == {"label": 13, "inferred": 5, "measured": 2}
    assert Counter(d.system for d in takeoff.ducts) == {"supply": 9, "return": 5, "unclassified": 6}

    sizes = [d.size for d in takeoff.ducts if d.source == "label"]
    for size, count in [(DuctSize(18), 1), (DuctSize(14), 2), (DuctSize(8), 2), (DuctSize(22, 14), 1)]:
        assert sizes.count(size) >= count

    grease = find([d.run for d in takeoff.ducts], 782, 456)
    duct = next(d for d in takeoff.ducts if d.run is grease)
    assert round(duct.length_ft * 12) == 226  # 18'-10"
    assert duct.system == "unclassified"

    dining = {d.system for d in takeoff.ducts if d.size == DuctSize(14) and d.run.length > 400}
    assert "supply" in dining and "return" not in dining
    assert {d.system for d in takeoff.ducts if d.size == DuctSize(22, 14)} == {"return"}
    # The 6" line from grille B/25 is one duct and must not be split between systems.
    assert {d.system for d in takeoff.ducts if round(d.width_in) == 6} == {"return"}
