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

    Only symbols touching a run end seed a system (risers and inline boxes); air devices hung off
    flex sit too far from their branch to attach reliably. Each run takes the system of the nearest
    seed in hops. A run no seed reaches, or one equally near a supply and a return seed, stays
    unclassified rather than guessed.
    """
    adj = {i: [j for j in js if not _through_symbol(runs[i], runs[j], terminals)] for i, js in neighbors(runs).items()}
    hops = {}
    for system in ("supply", "return"):
        seeds = [
            i for i, r in enumerate(runs)
            if any(t.system == system and min(t.distance(r.p0), t.distance(r.p1)) <= max(touch, 0.25 * r.width) for t in terminals)
        ]
        hops[system] = _hops(adj, seeds)

    out = []
    for i in range(len(runs)):
        s, r = hops["supply"].get(i, np.inf), hops["return"].get(i, np.inf)
        out.append("supply" if s < r else "return" if r < s else "unclassified")
    return out


def _hops(adj: dict[int, list[int]], seeds: list[int]) -> dict[int, int]:
    dist = dict.fromkeys(seeds, 0)
    queue = deque(seeds)
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in dist:
                dist[j] = dist[i] + 1
                queue.append(j)
    return dist


def _through_symbol(a: Run, b: Run, terminals: list[Terminal], inset: float = 1.0) -> bool:
    """True if the shortest link between two runs crosses a riser or device box.

    A duct ends at such a box, so two runs on either side of one are not joined to each other.
    """
    ends = [(p, b) for p in (a.p0, a.p1)] + [(p, a) for p in (b.p0, b.p1)]
    p, other = min(ends, key=lambda e: e[1].distance(e[0]))
    p = np.asarray(p)
    q = other.closest(p)
    pts = p + np.linspace(0, 1, int(np.hypot(*(q - p))) + 2)[:, None] * (q - p)
    for t in terminals:
        x0, y0, x1, y1 = t.box
        inside = (pts[:, 0] > x0 + inset) & (pts[:, 0] < x1 - inset) & (pts[:, 1] > y0 + inset) & (pts[:, 1] < y1 - inset)
        if inside.any():
            return True
    return False
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
