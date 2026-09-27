from collections import Counter, deque
from dataclasses import dataclass

import pymupdf

from .classify import assign_systems, find_terminals
from .geometry import Run, merge_collinear, neighbors, overlap, pair_walls, side_by_side
from .labels import DuctSize, has_size_mark, infer_scale, parse_scale, pick, size_readings
from .ocr import TextBox, ocr_vector_text, page_words
from .vector import dark_segments

BESIDE = 12.0  # pt; how far outside a duct wall a size label may sit


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
    pt_per_inch: float


def detect(page: pymupdf.Page, scale: str | None = None) -> Takeoff:
    segs = dark_segments(page)
    runs = merge_collinear(pair_walls(segs))

    boxes = [b for b in page_words(page) if size_readings(b.text)]
    if sum(has_size_mark(b.text) for b in boxes) < 2:
        # Sizes are not in the text layer (SHX text is drawn as strokes); OCR the rendered page,
        # which also covers whatever the text layer holds.
        def near_run(c):
            return any(r.distance(c) <= r.width / 2 + BESIDE for r in runs)

        boxes = [b for b in ocr_vector_text(page, segs, wanted=near_run) if size_readings(b.text)]

    if scale:
        scale_name, ppi = scale, parse_scale(scale)
    else:
        inside = [(size_readings(b.text), r.width) for b in boxes for r in runs if r.distance(b.center) <= r.width / 2]
        scale_name, ppi = infer_scale(inside)

    labelled = _assign_labels(runs, boxes, ppi)
    if not labelled:
        raise ValueError(f"no duct size label matches its duct width at {scale_name}; check --scale or omit it")
    accepted = _grow(runs, labelled, ppi)

    ducts = []
    order = sorted(accepted, key=lambda i: (round(runs[i].midpoint[1] / 20), runs[i].midpoint[0]))
    for n, i in enumerate(order, 1):
        run = runs[i]
        size, source, text = accepted[i]
        ducts.append(Duct(f"D{n}", run, size, source, text, run.width / ppi, run.length / ppi / 12))

    systems = assign_systems([d.run for d in ducts], find_terminals(segs))
    for duct, system in zip(ducts, systems):
        duct.system = system
    return Takeoff(ducts, scale_name, ppi)


def _assign_labels(runs: list[Run], boxes: list[TextBox], ppi: float) -> dict[int, tuple[DuctSize, str]]:
    """Give each size label to the nearest run whose measured width agrees with it."""
    votes: dict[int, list[tuple[DuctSize, str]]] = {}
    for box in boxes:
        readings = size_readings(box.text)
        reach = BESIDE if has_size_mark(box.text) else 0.0
        for gap, i in sorted((r.distance(box.center) - r.width / 2, i) for i, r in enumerate(runs)):
            if gap > reach:
                break
            if size := pick(readings, runs[i].width / ppi):
                votes.setdefault(i, []).append((size, box.text))
                break
    return {i: Counter(v).most_common(1)[0][0] for i, v in votes.items()}


def _grow(runs: list[Run], labelled: dict[int, tuple[DuctSize, str]], ppi: float) -> dict[int, tuple]:
    """Extend confirmed runs to connected runs that carry no label of their own.

    Same-width neighbours inherit the size first. Only when nothing more can be inherited is a
    width change accepted, and then only for runs that look like the confirmed ductwork: same
    lineweight, long enough, not a flex rib stack, not the gap between or a piece inside
    accepted ducts.
    """
    accepted = {i: (size, "label", text) for i, (size, text) in labelled.items()}
    weights = {round(runs[i].weight, 2) for i in accepted}
    adj = neighbors(runs)
    stacked = _stacked(runs)
    queue = deque(accepted)
    while queue:
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if j not in accepted and _same_width(runs[i], runs[j]) and not _covered(runs[j], runs, accepted):
                    size = accepted[i][0]
                    accepted[j] = (size, "inferred" if size else "measured", "")
                    queue.append(j)
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
                queue.append(j)
    return accepted


def _same_width(a: Run, b: Run) -> bool:
    return abs(a.width - b.width) <= max(1.0, 0.08 * max(a.width, b.width))


def _covered(run: Run, runs: list[Run], accepted: dict[int, tuple]) -> bool:
    owned = frozenset().union(*(runs[k].walls for k in accepted))
    if run.walls <= owned:
        return True
    return any(runs[k].distance(run.midpoint) < runs[k].width / 2 and runs[k].width > run.width for k in accepted)


def _stacked(runs: list[Run]) -> set[int]:
    """Runs sharing a wall with an overlapping parallel run: flex ribs and hatching, not ducts."""
    by_wall: dict[int, list[int]] = {}
    for i, r in enumerate(runs):
        for w in r.walls:
            by_wall.setdefault(w, []).append(i)
    out = set()
    for members in by_wall.values():
        for i in members:
            for j in members:
                a, b = runs[i], runs[j]
                if i < j and side_by_side(a, b) and overlap(a, b) >= 0.5 * max(a.length, b.length):
                    out.update((i, j))
    return out
