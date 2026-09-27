from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import Run, neighbors


@dataclass(frozen=True)
class Terminal:
    """An air device or riser symbol: a box with one diagonal (return) or an X (supply)."""

    box: tuple[float, float, float, float]
    system: str

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    def distance(self, pt) -> float:
        x0, y0, x1, y1 = self.box
        dx = max(x0 - pt[0], 0, pt[0] - x1)
        dy = max(y0 - pt[1], 0, pt[1] - y1)
        return float(np.hypot(dx, dy))


def find_terminals(segs: np.ndarray, min_side: float = 12.0, max_side: float = 80.0, tol: float = 1.5) -> list[Terminal]:
    p0, p1 = segs[:, 0:2], segs[:, 2:4]
    d = np.abs(p1 - p0)
    diagonal = (d.min(axis=1) >= min_side) & (d.max(axis=1) <= max_side) & (d.min(axis=1) >= 0.25 * d.max(axis=1))

    found: dict[tuple, Terminal] = {}
    for x0, y0, x1, y1, _ in segs[diagonal]:
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        key = tuple(round(v) for v in box)
        if key in found or not _has_box(segs, box, tol):
            continue
        bx0, by0, bx1, by1 = box
        other = ((bx0, by1), (bx1, by0)) if (x1 - x0) * (y1 - y0) > 0 else ((bx0, by0), (bx1, by1))
        crossed = _has_line(segs, *other, tol)
        found[key] = Terminal(box, "supply" if crossed else "return")
    return list(found.values())


def assign_systems(runs: list[Run], terminals: list[Terminal], touch: float = 3.0) -> list[str]:
    """Label runs supply/return from the symbols they physically meet, spread through the duct graph.

    Only symbols touching a run end count (risers and inline boxes). Air devices hung off flex sit
    too far from their branch to attach reliably, so runs no touching symbol reaches stay
    unclassified rather than guessed.
    """
    seeds: dict[int, tuple[float, str]] = {}
    for t in terminals:
        for i, r in enumerate(runs):
            dist = min(t.distance(r.p0), t.distance(r.p1))
            if dist <= max(touch, 0.25 * r.width) and (i not in seeds or dist < seeds[i][0]):
                seeds[i] = (dist, t.system)
    system = {i: s for i, (_, s) in seeds.items()}
    adj = neighbors(runs)
    queue = deque(system)
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in system:
                system[j] = system[i]
                queue.append(j)
    return [system.get(i, "unclassified") for i in range(len(runs))]


def _has_box(segs: np.ndarray, box, tol: float) -> bool:
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return all(_has_line(segs, a, b, tol) for a, b in zip(corners, corners[1:] + corners[:1]))


def _has_line(segs: np.ndarray, a, b, tol: float, cover: float = 0.8) -> bool:
    """True if collinear segments cover at least `cover` of the line a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.hypot(*(b - a))
    u = (b - a) / length
    n = np.array([-u[1], u[0]])
    rel0, rel1 = segs[:, 0:2] - a, segs[:, 2:4] - a
    on_line = (np.abs(rel0 @ n) <= tol) & (np.abs(rel1 @ n) <= tol)
    s0, s1 = rel0 @ u, rel1 @ u
    lo = np.clip(np.minimum(s0, s1), 0, length)[on_line]
    hi = np.clip(np.maximum(s0, s1), 0, length)[on_line]
    covered, reach = 0.0, 0.0
    for l, h in sorted(zip(lo, hi)):
        if h > reach:
            covered += h - max(l, reach)
            reach = h
    return covered >= cover * length
