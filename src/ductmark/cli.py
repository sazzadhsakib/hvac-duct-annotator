import argparse
import sys
from pathlib import Path

from .annotate import annotate, describe, save
from .pipeline import detect
from .vector import load_page


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ductmark", description="Detect, size and measure ductwork on an HVAC drawing PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory (default: out)")
    parser.add_argument("--page", type=int, default=0, help="0-based page index (default: 0)")
    parser.add_argument("--scale", help="drawing scale such as 1/4\"=1'-0\"; inferred from duct labels if omitted")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150)")
    args = parser.parse_args(argv)

    doc, page = load_page(str(args.pdf), args.page)
    try:
        takeoff = detect(page, args.scale)
    except ValueError as e:
        sys.exit(f"ductmark: {e}")

    annotate(page, takeoff)
    outputs = save(doc, page, takeoff, args.out, args.pdf.stem, args.dpi)

    print(f"scale {takeoff.scale}, {len(takeoff.ducts)} duct runs")
    for d in takeoff.ducts:
        print(f"  {describe(d):<28} {d.system:<13} {d.source}")
    for path in outputs:
        print(f"wrote {path}")
