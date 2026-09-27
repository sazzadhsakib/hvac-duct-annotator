import numpy as np
import pymupdf

# MEP work is drawn dark over a screened-grey architectural background.
MAX_LUMA = 0.25


def load_page(path: str, index: int = 0) -> tuple[pymupdf.Document, pymupdf.Page]:
    doc = pymupdf.open(path)
    page = doc[index]
    # Drawings come back unrotated while renders are rotated; baking the rotation
    # into the content puts extraction, OCR and overlays in one coordinate frame.
    page.remove_rotation()
    return doc, page


def dark_segments(page: pymupdf.Page, max_luma: float = MAX_LUMA, min_len: float = 9.0) -> np.ndarray:
    """Straight stroked edges as rows of [x0, y0, x1, y1, lineweight] in page points."""
    rows = []
    for path in page.get_drawings():
        color = path.get("color")
        if color is None or _luma(color) > max_luma:
            continue
        weight = path.get("width") or 0.0
        for item in path["items"]:
            for a, b in _edges(item):
                rows.append((a.x, a.y, b.x, b.y, weight))

    segs = np.array(rows, dtype=float).reshape(-1, 5)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return _dedupe(segs[lengths >= min_len])


def _luma(color) -> float:
    if len(color) == 1:
        return color[0]
    if len(color) == 4:
        color = [(1 - c) * (1 - color[3]) for c in color[:3]]
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edges(item) -> list:
    kind = item[0]
    if kind == "l":
        return [(item[1], item[2])]
    if kind == "re":
        quad = item[1].quad
    elif kind == "qu":
        quad = item[1]
    else:
        return []
    corners = [quad.ul, quad.ur, quad.lr, quad.ll]
    return list(zip(corners, corners[1:] + corners[:1]))


def _dedupe(segs: np.ndarray) -> np.ndarray:
    flip = (segs[:, 0] > segs[:, 2]) | ((segs[:, 0] == segs[:, 2]) & (segs[:, 1] > segs[:, 3]))
    segs[flip] = segs[flip][:, [2, 3, 0, 1, 4]]
    _, first = np.unique(np.round(segs[:, :4], 1), axis=0, return_index=True)
    return segs[np.sort(first)]
