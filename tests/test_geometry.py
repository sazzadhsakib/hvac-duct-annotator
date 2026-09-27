import math

import numpy as np
import pytest

from ductmark.geometry import pair_walls


def segs(*lines, weight=1.0):
    return np.array([[*a, *b, weight] for a, b in lines], dtype=float)


def rect(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def test_rectangle_becomes_one_piece():
    (piece,) = pair_walls(segs(*rect(0, 0, 200, 12)))
    assert piece.width == pytest.approx(12)
    assert piece.length == pytest.approx(200)
    assert piece.angle == pytest.approx(0)


def test_transition_pairs_continuous_wall_twice():
    lines = [((0, 0), (300, 0)), ((0, 12), (120, 12)), ((150, 18), (300, 18))]
    pieces = sorted(pair_walls(segs(*lines)), key=lambda p: p.width)
    assert [round(p.width) for p in pieces] == [12, 18]
    assert [round(p.length) for p in pieces] == [120, 150]


def test_outer_walls_of_adjacent_ducts_do_not_pair():
    lines = [((0, y), (200, y)) for y in (0, 12, 20, 32)]
    widths = sorted(round(p.width) for p in pair_walls(segs(*lines)))
    assert widths == [8, 12, 12]


def test_liner_close_to_wall_does_not_block():
    lines = [((0, 0), (200, 0)), ((0, 2), (200, 2)), ((0, 30), (200, 30))]
    assert 30 in {round(p.width) for p in pair_walls(segs(*lines))}


@pytest.mark.parametrize(
    "lines",
    [
        [((x, 0), (x, 10)) for x in range(0, 40, 4)],
        rect(0, 0, 24, 24),
    ],
    ids=["flex-ribs", "square-symbol"],
)
def test_short_overlaps_are_ignored(lines):
    assert pair_walls(segs(*lines)) == []


def test_diagonal_duct():
    c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
    corners = [(0, 0), (150, 0), (150, 18), (0, 18)]
    pts = [(x * c - y * s, x * s + y * c) for x, y in corners]
    (piece,) = pair_walls(segs(*zip(pts, pts[1:] + pts[:1])))
    assert piece.angle == pytest.approx(45)
    assert piece.width == pytest.approx(18)
    assert piece.length == pytest.approx(150)
