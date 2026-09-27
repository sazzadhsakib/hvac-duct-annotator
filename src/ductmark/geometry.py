from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Run:
    """A straight duct piece: centerline p0->p1 in page points, wall-to-wall width in points."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    width: float
    weight: float
    walls: frozenset[int]

    @property
    def length(self) -> float:
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    @property
    def direction(self) -> np.ndarray:
        return np.subtract(self.p1, self.p0) / self.length

    @property
    def normal(self) -> np.ndarray:
        ux, uy = self.direction
        return np.array([-uy, ux])

    @property
    def angle(self) -> float:
        ux, uy = self.direction
        return float(np.degrees(np.arctan2(uy, ux)) % 180)

    @property
    def midpoint(self) -> np.ndarray:
        return (np.asarray(self.p0) + np.asarray(self.p1)) / 2

    def closest(self, pt) -> np.ndarray:
        t = np.clip(np.subtract(pt, self.p0) @ self.direction, 0, self.length)
        return np.asarray(self.p0) + t * self.direction

    def distance(self, pt) -> float:
        return float(np.hypot(*np.subtract(pt, self.closest(pt))))


def pair_walls(
    segs: np.ndarray,
    min_gap: float = 4.0,
    max_gap: float = 80.0,
    angle_tol: float = 1.5,
    min_overlap: float = 18.0,
    block_cover: float = 0.5,
) -> list[Run]:
    """Pair near-parallel wall segments into duct pieces.

    A pair spans only the interval where both walls overlap, so one long wall can pair with
    several shorter opposite walls (transitions, tees). Short overlaps relative to the gap are
    rejected (flex ribs, grille louvers, symbol boxes), as are pairs with another parallel wall
    between them (the outer walls of two adjacent ducts).
    """
    p0, p1 = segs[:, 0:2], segs[:, 2:4]
    length = np.hypot(*(p1 - p0).T)
    u = (p1 - p0) / length[:, None]
    n = np.stack([-u[:, 1], u[:, 0]], axis=1)
    ang = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 180
    idx = np.arange(len(segs))

    pieces = []
    for start in range(0, len(segs), 512):
        rows = idx[start:start + 512]
        dang = np.abs(ang[rows, None] - ang[None, :])
        parallel = np.minimum(dang, 180 - dang) <= angle_tol
        rel0 = p0[None] - p0[rows, None]
        rel1 = p1[None] - p0[rows, None]
        off = ((rel0 + rel1) / 2 * n[rows, None]).sum(-1)
        s0 = (rel0 * u[rows, None]).sum(-1)
        s1 = (rel1 * u[rows, None]).sum(-1)
        lo = np.maximum(0, np.minimum(s0, s1))
        hi = np.minimum(length[rows, None], np.maximum(s0, s1))
        gap = np.abs(off)
        ok = (
            parallel
            & (idx[None] > rows[:, None])
            & (gap >= min_gap)
            & (gap <= max_gap)
            & (hi - lo >= np.maximum(2 * gap, min_overlap))
        )
        for r, j in zip(*np.nonzero(ok)):
            i = rows[r]
            # Another wall strictly between the pair (liners hugging a wall excepted) blocks it.
            side = off[r] * np.sign(off[r, j])
            between = (side > min_gap) & (side < gap[r, j] - min_gap)
            cover = np.minimum(hi[r, j], np.maximum(s0[r], s1[r])) - np.maximum(lo[r, j], np.minimum(s0[r], s1[r]))
            if np.any(parallel[r] & between & (cover >= block_cover * (hi[r, j] - lo[r, j]))):
                continue
            mid = off[r, j] / 2
            a = p0[i] + u[i] * lo[r, j] + n[i] * mid
            b = p0[i] + u[i] * hi[r, j] + n[i] * mid
            pieces.append(_run(a, b, gap[r, j], max(segs[i, 4], segs[j, 4]), {int(i), int(j)}))
    return pieces


def merge_collinear(
    pieces: list[Run], width_tol: float = 0.08, gap_factor: float = 3.0, angle_tol: float = 1.5
) -> list[Run]:
    """Join same-width pieces on one axis that are split by dampers, risers or branch openings."""
    parent = list(range(len(pieces)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            if _mergeable(a, pieces[j], width_tol, gap_factor, angle_tol):
                parent[find(i)] = find(j)

    groups: dict[int, list[Run]] = {}
    for i, piece in enumerate(pieces):
        groups.setdefault(find(i), []).append(piece)
    return [_combine(g) for g in groups.values()]


def neighbors(runs: list[Run], reach: float = 2.0, angle_tol: float = 10.0) -> dict[int, list[int]]:
    """Runs joined by a fitting (elbow, tee, transition).

    A run end connects to another run when the gap from that end to the other run's wall is at
    most `reach` times the approaching run's own width; a branch stopping short of a wide trunk
    is not joined to it.
    """
    adj: dict[int, list[int]] = {i: [] for i in range(len(runs))}
    for i, a in enumerate(runs):
        for j in range(i + 1, len(runs)):
            b = runs[j]
            # Side-by-side parallel runs are not connected; only end-to-end jogs are.
            if side_by_side(a, b, angle_tol):
                continue
            ends = [(p, a, b) for p in (a.p0, a.p1)] + [(p, b, a) for p in (b.p0, b.p1)]
            if any(other.distance(p) - other.width / 2 <= reach * own.width for p, own, other in ends):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def side_by_side(a: Run, b: Run, angle_tol: float = 10.0) -> bool:
    return _parallel(a, b, angle_tol) and _axial_gap(a, b) < 0


def _mergeable(a: Run, b: Run, width_tol: float, gap_factor: float, angle_tol: float) -> bool:
    w = max(a.width, b.width)
    if abs(a.width - b.width) > max(1.0, width_tol * w) or not _parallel(a, b, angle_tol):
        return False
    if abs((b.midpoint - np.asarray(a.p0)) @ a.normal) > 0.25 * w:
        return False
    return _axial_gap(a, b) <= gap_factor * w


def _parallel(a: Run, b: Run, tol: float) -> bool:
    d = abs(a.angle - b.angle)
    return min(d, 180 - d) <= tol


def overlap(a: Run, b: Run) -> float:
    """Length of b's projection that falls within a."""
    s0, s1 = _span(a, b)
    return max(0.0, min(a.length, s1) - max(0.0, s0))


def _axial_gap(a: Run, b: Run) -> float:
    s0, s1 = _span(a, b)
    return max(s0 - a.length, -s1)


def _span(a: Run, b: Run) -> list[float]:
    return sorted(float(np.subtract(p, a.p0) @ a.direction) for p in (b.p0, b.p1))


def _combine(group: list[Run]) -> Run:
    if len(group) == 1:
        return group[0]
    axis = max(group, key=lambda r: r.length)
    origin, u, n = np.asarray(axis.p0), axis.direction, axis.normal
    s = [np.subtract(p, origin) @ u for r in group for p in (r.p0, r.p1)]
    lengths = np.array([r.length for r in group])
    off = np.average([(r.midpoint - origin) @ n for r in group], weights=lengths)
    width = np.average([r.width for r in group], weights=lengths)
    walls = frozenset().union(*(r.walls for r in group))
    return _run(origin + u * min(s) + n * off, origin + u * max(s) + n * off, width, max(r.weight for r in group), walls)


def _run(a, b, width: float, weight: float, walls) -> Run:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    if d[0] < 0 or (d[0] == 0 and d[1] < 0):
        a, b = b, a
    return Run(tuple(map(float, a)), tuple(map(float, b)), float(width), float(weight), frozenset(walls))
