from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pymupdf

from .classify import assign_systems, find_terminals
from .geometry import Run, merge_collinear, neighbors, overlap, pair_walls, side_by_side
from .labels import MAX_SIZE, MIN_SIZE, DuctSize, has_size_mark, infer_scale, parse_scale, pick, scale_notes, size_readings
from .ocr import TextBox, ocr_vector_text, page_words
from .vector import dark_segments

BESIDE = 12.0  # pt; how far outside a duct wall a size label may sit
AMBIGUOUS = 2.0  # pt; runs whose fit to a label differs by less than this are indistinguishable
# Real sizes in inches, converted to paper with the drawing scale once it is known.
MIN_STRAIGHT = 12  # shortest straight run to pair
SYMBOL_SIDES = (8, 54)  # riser and air-device boxes


@dataclass
class Duct:
    id: str
    run: Run
    size: DuctSize | None
    source: str  # "label" | "inferred" | "measured"
    label: str
    width_in: float
    length_ft: float
    system: str = "unclassified"


@dataclass
class Takeoff:
    ducts: list[Duct]
    scale: str


def detect(page: pymupdf.Page, scale: str | None = None) -> Takeoff:
    segs = dark_segments(page)
    limits = {}
    if scale:
        # Pairing precedes scale inference, so only a given scale can set the duct width range;
        # otherwise the paper-point defaults (about 2.7"-53" at 1/4"=1'-0") apply.
        ppi = parse_scale(scale)
        limits = {"min_gap": 0.9 * MIN_SIZE * ppi, "max_gap": 1.1 * MAX_SIZE * ppi, "min_overlap": MIN_STRAIGHT * ppi}
    runs = merge_collinear(pair_walls(segs, **limits))

    words = page_words(page)
    if sum(has_size_mark(w.text) for w in words if size_readings(w.text)) < 2:
        # Sizes are not in the text layer (SHX text is drawn as strokes); OCR the rendered page,
        # which also covers whatever the text layer holds.
        def near_run(c):
            return any(r.distance(c) <= r.width / 2 + BESIDE for r in runs)

        words = ocr_vector_text(page, segs, wanted=near_run)

    boxes = [w for w in words if size_readings(w.text)]
    marked = [b for b in boxes if has_size_mark(b.text)]
    bare = [b for b in boxes if not has_size_mark(b.text)]

    if scale:
        scale_name, ppi = scale, parse_scale(scale)
    else:
        # Ceiling grids, hatching and flex ribs hold text by accident; only labels in duct-like
        # runs vote.
        stacked = _stacked(runs, same_width=True)
        inside = []
        for b in marked:
            gap, i = min((r.distance(b.center) - r.width / 2, i) for i, r in enumerate(runs))
            if gap <= 0 and i not in stacked:
                inside.append((size_readings(b.text), runs[i].width))
        scale_name, ppi = infer_scale(inside, scale_notes([page.get_text(), *(w.text for w in words)]))

    adj = neighbors(runs)
    labelled = _assign_labels(runs, marked, ppi)
    if not labelled:
        raise ValueError(f"no duct size label matches its duct width at {scale_name}; check --scale or omit it")
    accepted = _grow(runs, adj, labelled, ppi)

    # A bare number is either a size whose ø and inch marks OCR lost, or an unrelated callout, CFM
    # or room number. It only counts inside a run no explicit label claims, and only on ductwork
    # the explicit labels already confirmed or on a run touching it.
    corroborated = {
        i: label for i, label in _assign_labels(runs, bare, ppi).items()
        if i not in labelled and (i in accepted or any(j in accepted for j in adj[i]))
    }
    if corroborated:
        accepted = _grow(runs, adj, labelled | corroborated, ppi)

    ducts = []
    order = sorted(accepted, key=lambda i: (round(runs[i].midpoint[1] / 20), runs[i].midpoint[0]))
    for n, i in enumerate(order, 1):
        run = runs[i]
        size, source, text = accepted[i]
        ducts.append(Duct(f"D{n}", run, size, source, text, run.width / ppi, run.length / ppi / 12))

    sides = [side * ppi for side in SYMBOL_SIDES]
    systems = assign_systems([d.run for d in ducts], find_terminals(segs, *sides))
    for duct, system in zip(ducts, systems):
        duct.system = system
    return Takeoff(ducts, scale_name)


def _assign_labels(runs: list[Run], boxes: list[TextBox], ppi: float) -> dict[int, tuple[DuctSize, str]]:
    """Give each size label to the run it sits in, or beside, whose measured width agrees with it.

    The label must lie within the run's length. A label that two runs fit about equally well, or a
    run holding different sizes in equal number, is left unassigned rather than guessed.
    """
    votes: dict[int, list[tuple[DuctSize, str]]] = {}
    for box in boxes:
        readings = size_readings(box.text)
        reach = BESIDE if has_size_mark(box.text) else 0.0
        fits = []
        for i, r in enumerate(runs):
            gap = r.distance(box.center) - r.width / 2
            along = np.subtract(box.center, r.p0) @ r.direction
            if gap <= reach and 0 <= along <= r.length and (size := pick(readings, r.width / ppi)):
                fits.append((gap, i, size))
        fits.sort(key=lambda f: f[0])
        if not fits or (len(fits) > 1 and fits[1][0] - fits[0][0] < AMBIGUOUS):
            continue
        _, i, size = fits[0]
        votes.setdefault(i, []).append((size, box.text))

    labelled = {}
    for i, v in votes.items():
        (size, n), *rest = Counter(s for s, _ in v).most_common()
        if not rest or rest[0][1] < n:
            labelled[i] = (size, next(text for s, text in v if s == size))
    return labelled


def _grow(runs: list[Run], adj: dict[int, list[int]], labelled: dict[int, tuple[DuctSize, str]], ppi: float) -> dict[int, tuple]:
    """Extend confirmed runs to connected runs that carry no label of their own.

    Same-width neighbours inherit the size first. Only when nothing more can be inherited is a
    width change admitted, so a run is never sized by a different-width neighbour it could have
    inherited from.
    """
    accepted = {i: (size, "label", text) for i, (size, text) in labelled.items()}
    queue = deque(accepted)
    while queue:
        _inherit_same_width(runs, adj, accepted, queue)
        queue.extend(_admit_width_changes(runs, adj, accepted, ppi))
    return accepted


def _inherit_same_width(runs: list[Run], adj: dict[int, list[int]], accepted: dict[int, tuple], queue: deque) -> None:
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if j not in accepted and _same_width(runs[i], runs[j]) and not _covered(runs[j], runs, accepted):
                size = accepted[i][0]
                accepted[j] = (size, "inferred" if size else "measured", "")
                queue.append(j)


def _admit_width_changes(runs: list[Run], adj: dict[int, list[int]], accepted: dict[int, tuple], ppi: float) -> list[int]:
    """Connected runs of another width that look like the confirmed ductwork: same lineweight,
    long enough, not a flex rib stack, not the gap between or a piece inside accepted ducts."""
    weights = {round(runs[i].weight, 2) for i, (_, source, _) in accepted.items() if source == "label"}
    stacked = _stacked(runs)
    admitted = []
    for j in dict.fromkeys(j for i in accepted for j in adj[i]):
        run = runs[j]
        if (
            j not in accepted
            and j not in stacked
            and round(run.weight, 2) in weights
            and run.length >= max(24 * ppi, 3 * run.width)
            and not _covered(run, runs, accepted)
        ):
            accepted[j] = (None, "measured", "")
            admitted.append(j)
    return admitted


def _same_width(a: Run, b: Run) -> bool:
    return abs(a.width - b.width) <= max(1.0, 0.08 * max(a.width, b.width))


def _covered(run: Run, runs: list[Run], accepted: dict[int, tuple]) -> bool:
    owned = frozenset().union(*(runs[k].walls for k in accepted))
    if run.walls <= owned:
        return True
    return any(runs[k].distance(run.midpoint) < runs[k].width / 2 and runs[k].width > run.width for k in accepted)


def _stacked(runs: list[Run], same_width: bool = False) -> set[int]:
    """Runs sharing a wall with an overlapping parallel run: flex ribs and hatching, not ducts.

    With same_width, only repeated cells of equal width count (ceiling grids, ribs), so a duct
    running beside another duct is not caught by the gap between them.
    """
    by_wall: dict[int, list[int]] = {}
    for i, r in enumerate(runs):
        for w in r.walls:
            by_wall.setdefault(w, []).append(i)
    out = set()
    for members in by_wall.values():
        for i in members:
            for j in members:
                a, b = runs[i], runs[j]
                if i < j and (not same_width or _same_width(a, b)) and side_by_side(a, b) and overlap(a, b) >= 0.5 * max(a.length, b.length):
                    out.update((i, j))
    return out
