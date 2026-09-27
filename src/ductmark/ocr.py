from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class TextBox:
    text: str
    center: tuple[float, float]
    height: float


def page_words(page: pymupdf.Page) -> list[TextBox]:
    boxes = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        boxes.append(TextBox(word, ((x0 + x1) / 2, (y0 + y1) / 2), min(x1 - x0, y1 - y0)))
    return boxes
