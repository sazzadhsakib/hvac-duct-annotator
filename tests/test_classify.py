import numpy as np

from ductmark.classify import Terminal, assign_systems, find_terminals
from ductmark.geometry import Run


def box_segments(x0, y0, x1, y1, diagonals):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    lines = list(zip(corners, corners[1:] + corners[:1]))
    if diagonals >= 1:
        lines.append(((x0, y0), (x1, y1)))
    if diagonals == 2:
        lines.append(((x0, y1), (x1, y0)))
    return [[*a, *b, 1.0] for a, b in lines]


def test_x_box_is_supply_and_single_diagonal_is_return():
    segs = np.array(box_segments(0, 0, 36, 36, 2) + box_segments(100, 0, 136, 36, 1))
    systems = {t.center: t.system for t in find_terminals(segs)}
    assert systems == {(18.0, 18.0): "supply", (118.0, 18.0): "return"}


def test_open_diagonal_without_box_is_ignored():
    segs = np.array([[0, 0, 36, 36, 1.0], [0, 0, 36, 0, 1.0]])
    assert find_terminals(segs) == []


def run(a, b, width=12.0):
    return Run(a, b, width, 1.0, frozenset())


def test_systems_spread_from_touching_symbols_only():
    runs = [
        run((0, 0), (100, 0)),  # starts at the supply riser
        run((118, 18), (118, 120)),  # elbow off run 0
        run((300, 0), (400, 0)),  # ends at the return riser
        run((600, 0), (700, 0)),  # near a diffuser, not touching
    ]
    terminals = [
        Terminal((-20, -10, 0, 10), "supply"),
        Terminal((400, -10, 420, 10), "return"),
        Terminal((710, -20, 746, 16), "supply"),
    ]
    assert assign_systems(runs, terminals) == ["supply", "supply", "return", "unclassified"]


def test_run_equally_near_supply_and_return_is_unclassified():
    runs = [
        run((0, 0), (100, 0)),  # starts at the supply riser
        run((118, 18), (118, 120)),  # one elbow from each
        run((136, 138), (236, 138)),  # ends at the return riser
    ]
    terminals = [Terminal((-20, -10, 0, 10), "supply"), Terminal((236, 128, 256, 148), "return")]
    assert assign_systems(runs, terminals) == ["supply", "unclassified", "return"]
    assert assign_systems(runs, terminals[::-1]) == ["supply", "unclassified", "return"]


def test_run_touching_both_kinds_of_symbol_is_unclassified():
    runs = [run((0, 0), (100, 0))]
    terminals = [Terminal((-20, -10, 0, 10), "supply"), Terminal((100, -10, 120, 10), "return")]
    assert assign_systems(runs, terminals) == ["unclassified"]


def test_runs_on_either_side_of_a_riser_box_are_not_joined():
    runs = [run((0, 0), (100, 0), 18), run((140, 10), (140, 100), 12)]
    supply = Terminal((-20, -10, 0, 10), "supply")
    riser = Terminal((106, -20, 122, 20), "return")
    assert assign_systems(runs, [supply]) == ["supply", "supply"]
    assert assign_systems(runs, [supply, riser]) == ["supply", "unclassified"]
