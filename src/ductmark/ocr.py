from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
import pymupdf

from .labels import size_readings

# Paper-space sizes in points. Annotation text on MEP sheets is ~3/32"-1/8" (6.75-9 pt) tall
# regardless of drawing scale, so these hold across sheets.
GLYPH_MAX = 14.0
GLYPH_MIN = 1.5
JOIN = 3.5
LONG_STROKE = 18.0


@dataclass(frozen=True)
class TextBox:
    text: str
    center: tuple[float, float]


def page_words(page: pymupdf.Page) -> list[TextBox]:
    boxes = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        boxes.append(TextBox(word, ((x0 + x1) / 2, (y0 + y1) / 2)))
    return boxes


def ocr_vector_text(
    page: pymupdf.Page,
    segs: np.ndarray,
    wanted: Callable[[tuple[float, float]], bool] | None = None,
    dpi: int = 300,
    ink_level: int = 70,
) -> list[TextBox]:
    """OCR text drawn as vector strokes (AutoCAD SHX fonts), which has no text layer.

    Long strokes are erased so glyphs no longer touch duct walls, glyph-sized ink blobs are
    joined into words, and each word is deskewed from its own minimum-area rectangle, so
    horizontal, vertical and diagonal labels are all read upright.
    """
    zoom = dpi / 72
    glyphs = _glyph_mask(_ink_mask(page, segs, zoom, ink_level), zoom)
    k = int(JOIN * zoom) | 1
    words = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(words, connectivity=8)

    boxes = []
    for i in range(1, n):
        center = (float(centroids[i][0] / zoom), float(centroids[i][1] / zoom))
        if wanted is not None and not wanted(center):
            continue
        x, y, w, h, _ = stats[i]
        word = (labels[y:y + h, x:x + w] == i) & (glyphs[y:y + h, x:x + w] > 0)
        crop = _deskew(word, pad=int(3 * zoom))
        if crop is not None and (text := _read(crop)):
            boxes.append(TextBox(text, center))
    return boxes


def _ink_mask(page: pymupdf.Page, segs: np.ndarray, zoom: float, ink_level: int) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.stride)[:, : pix.w]
    # Below the lightest screened background gray, so architecture drops out.
    mask = np.where(gray < ink_level, 255, 0).astype(np.uint8)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    for x0, y0, x1, y1, weight in segs[lengths >= LONG_STROKE]:
        a = (round(x0 * zoom), round(y0 * zoom))
        b = (round(x1 * zoom), round(y1 * zoom))
        cv2.line(mask, a, b, 0, max(2, round(weight * zoom) + 1))
    return mask


def _glyph_mask(mask: np.ndarray, zoom: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    w, h = stats[:, 2], stats[:, 3]
    keep = (np.maximum(w, h) <= GLYPH_MAX * zoom) & (np.maximum(w, h) >= GLYPH_MIN * zoom)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _deskew(word: np.ndarray, pad: int) -> np.ndarray | None:
    pts = np.column_stack(np.nonzero(word)[::-1]).astype(np.float32)
    if len(pts) < 10:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w < h:
        w, h, angle = h, w, angle + 90
    angle = (angle + 90) % 180 - 90
    src = cv2.copyMakeBorder(np.where(word, 0, 255).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    cx, cy = cx + pad, cy + pad
    out_w, out_h = int(w) + 2 * pad, int(h) + 2 * pad
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[:, 2] += (out_w / 2 - cx, out_h / 2 - cy)
    return cv2.warpAffine(src, m, (out_w, out_h), borderValue=255)


def _read(crop: np.ndarray) -> str:
    text = _recognize(crop, use_cls=True)
    if size_readings(text):
        return text
    # The angle classifier misses some upside-down vertical labels.
    flipped = _recognize(cv2.rotate(crop, cv2.ROTATE_180), use_cls=False)
    return flipped if size_readings(flipped) else text


def _recognize(crop: np.ndarray, use_cls: bool) -> str:
    result, _ = _engine()(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), use_det=False, use_cls=use_cls, use_rec=True)
    return result[0][0].strip() if result else ""


@cache
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(text_score=0.3)
