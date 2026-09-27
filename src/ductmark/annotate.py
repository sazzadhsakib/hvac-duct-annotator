import csv
from pathlib import Path

import pymupdf

from .labels import format_ft_in
from .pipeline import Duct, Takeoff

COLORS = {
    "supply": (0.05, 0.35, 0.95),
    "return": (0.9, 0.15, 0.1),
    "unclassified": (0.35, 0.35, 0.35),
}
TAG_SIZE = 6.5


def describe(duct: Duct) -> str:
    size = str(duct.size) if duct.size else f'~{round(duct.width_in)}"'
    return f"{duct.id}  {size}  {format_ft_in(duct.length_ft)}"


def annotate(page: pymupdf.Page, takeoff: Takeoff) -> None:
    for duct in takeoff.ducts:
        color = COLORS[duct.system]
        width = max(3.0, 0.6 * duct.run.width)
        measured = duct.source == "measured"
        shape = page.new_shape()
        shape.draw_line(duct.run.p0, duct.run.p1)
        shape.finish(
            color=color,
            width=width,
            stroke_opacity=0.45,
            # Round caps would fill the gaps of a dashed stroke this wide.
            lineCap=0 if measured else 1,
            dashes=f"[{2 * width:.1f} {width:.1f}] 0" if measured else None,
        )
        shape.commit()
        _tag(page, duct, color)
    _legend(page, takeoff)


def _tag(page: pymupdf.Page, duct: Duct, color) -> None:
    text = describe(duct)
    width = pymupdf.get_text_length(text, fontname="helv", fontsize=TAG_SIZE)
    normal = duct.run.normal if duct.run.normal[1] <= 0 else -duct.run.normal
    # Tags are horizontal, so their extent along the normal depends on the run's direction.
    extent = abs(normal[0]) * width / 2 + abs(normal[1]) * TAG_SIZE / 2
    anchor = duct.run.midpoint + normal * (duct.run.width / 2 + 3 + extent)
    box = pymupdf.Rect(anchor[0] - width / 2 - 1.5, anchor[1] - TAG_SIZE / 2 - 1.5, anchor[0] + width / 2 + 1.5, anchor[1] + TAG_SIZE / 2 + 1.5)
    page.draw_rect(box, color=color, fill=(1, 1, 1), width=0.6, fill_opacity=0.85)
    page.insert_text((box.x0 + 1.5, box.y1 - 2.8), text, fontname="helv", fontsize=TAG_SIZE, color=color)


def _legend(page: pymupdf.Page, takeoff: Takeoff) -> None:
    lines = [f"Ducts detected: {len(takeoff.ducts)}   scale {takeoff.scale}"]
    for system, color in COLORS.items():
        members = [d for d in takeoff.ducts if d.system == system]
        if members:
            total = format_ft_in(sum(d.length_ft for d in members))
            lines.append((f"{system}: {len(members)} runs, {total}", color))
    lines.append("dashed = size not labelled (measured width)")

    x, y = page.rect.x0 + 18, page.rect.y0 + 18
    height = 10 * len(lines) + 6
    page.draw_rect(pymupdf.Rect(x, y, x + 220, y + height), color=(0, 0, 0), fill=(1, 1, 1), width=0.6)
    for n, line in enumerate(lines):
        text, color = line if isinstance(line, tuple) else (line, (0, 0, 0))
        page.insert_text((x + 5, y + 12 + 10 * n), text, fontname="helv", fontsize=7.5, color=color)


def save(doc: pymupdf.Document, page: pymupdf.Page, takeoff: Takeoff, out_dir: Path, stem: str, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png, report = (out_dir / f"{stem}_annotated.pdf", out_dir / f"{stem}_annotated.png", out_dir / f"{stem}_ducts.csv")
    doc.save(pdf, garbage=3, deflate=True)
    page.get_pixmap(dpi=dpi).save(png)
    with report.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "system", "size", "shape", "width_in", "length_ft", "length", "source", "label_text", "x0", "y0", "x1", "y1"])
        for d in takeoff.ducts:
            writer.writerow([
                d.id, d.system, str(d.size) if d.size else "", d.size.shape if d.size else "",
                round(d.width_in, 1), round(d.length_ft, 2), format_ft_in(d.length_ft), d.source, d.label,
                *(round(v, 1) for v in (*d.run.p0, *d.run.p1)),
            ])
    return [pdf, png, report]
