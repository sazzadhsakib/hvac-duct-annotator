# hvac-duct-annotator

> [!IMPORTANT]
> **Demo video (8 min):** [ductmark-demo.mp4](https://github.com/sazzadhsakib/hvac-duct-annotator/releases/download/v0.1.0/ductmark-demo.mp4), about 35 MB. It shows a live run on the sample drawing, the annotated result, and results on three other public drawings.

`ductmark` reads an HVAC mechanical plan PDF. For each duct run it finds the run, reads its size label and measures its length from the drawing scale. It writes three files:
- an annotated PDF
- a PNG of the same page
- a CSV takeoff

![Annotated sample](docs/sample_output.png)

Colour shows the system: blue is supply, red is return, grey is unclassified. Each tag shows the run id, its size and its straight length. A dashed line means the size was measured from the drawing, because the run has no size label.

## Setup

Requires Python 3.11–3.12 (`rapidocr-onnxruntime` declares `<3.13`) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

With plain pip: `pip install .` in a 3.11–3.12 environment.

## Usage

```bash
uv run ductmark samples/testset2.pdf -o out
uv run ductmark drawing.pdf --page 2 --scale "1/8\"=1'-0\""
```

| Option | Default | |
|---|---|---|
| `-o, --out` | `out` | output directory |
| `--page` | `0` | 0-based page index |
| `--scale` | inferred | e.g. `1/4"=1'-0"`, `1"=20'`, `1:50` |
| `--dpi` | `150` | PNG resolution |
| `--max-luma` | `0.25` | lightest stroke (0–1) read as duct geometry; raise it (e.g. `0.35`) for ducts drawn in colour |

Outputs, for an input named `<stem>.pdf`:

- `<stem>_annotated.pdf`: the original sheet with the overlays and a legend.
- `<stem>_annotated.png`: a render of the same page.
- `<stem>_ducts.csv`:
  - `id`, `system`, `size`, `shape`, `width_in`, `length_ft`, `length`
  - `source`: `label`, `inferred` or `measured`
  - `label_text`: the raw OCR string
  - the run's centerline in page points

## How it works

The sample is a Bluebeam-flattened AutoCAD export. Its duct walls are exact vector lines. The size labels (`12"ø`, `22"X14"`) and the scale are SHX glyphs drawn as strokes, so they are not in the text layer. The pipeline therefore takes geometry from the vectors and reads text with OCR, and each check confirms the other.

1. **Segments** (`vector.py`)
   - Rotation is baked into the page first, so extraction, rendering and overlays share one coordinate frame.
   - Straight stroked edges come from lines, rectangles and quads.
   - Only dark strokes are kept (luminance ≤ 0.25, adjustable with `--max-luma`), because MEP work is drawn dark over a screened-grey architectural background.
2. **Walls to runs** (`geometry.py`)
   - Near-parallel segments are paired over the interval where they overlap. One long wall can therefore pair with several opposite walls, which handles transitions and tees.
   - With `--scale`, walls may be 3"–60" apart at that scale. Otherwise the limits are 4–80 pt of paper (about 2.7"–53" at 1/4"=1'-0"), because the scale is inferred later from the runs.
   - These pairs are rejected:
     - overlaps shorter than twice the gap: flex ribs, grille louvres, symbol boxes;
     - pairs with another parallel wall between them: the outer walls of two adjacent ducts.
   - Collinear pieces of the same width are merged across dampers and branch openings.
3. **Reading sizes** (`ocr.py`, `labels.py`)
   - The PDF text layer is used when it contains size labels. Otherwise the page is OCR'd at 300 dpi:
     - long strokes are erased so glyphs no longer touch duct walls;
     - glyph-sized blobs are joined into words;
     - each word is deskewed from its minimum-area rectangle, so horizontal, vertical and diagonal labels are all read upright;
     - RapidOCR's recogniser runs on each word, without its detector.
   - The parser accepts the ways OCR misreads SHX text. The ø can come back as `0`, `o` or `g`, and the inch mark can be dropped. The measured wall gap decides between readings: `120` inside a duct 12" wide is `12"ø`.
4. **Scale**: taken from `--scale`, or inferred. Inference tries each standard architectural and engineering scale and counts, for each, how many explicit labels match the wall gap of the run they sit in.
   - Labels that sit in repeated equal cells (a ceiling grid, hatching, flex ribs) don't vote; they fit those cells by accident.
   - A scale written on the sheet, such as the title block's `1/4"=1'-0"`, must be among the best supported and have two agreeing labels, and then settles a tie.
   - Without one, the winner needs three agreeing labels and twice the votes of the runner-up.
   - Anything less stops with an error asking for `--scale`.
   - On the sample, 11 labels agree at 1/4"=1'-0" (1.5 pt per real inch), 1 fits 3/8", and the title block's scale agrees.
5. **Confirm and grow** (`pipeline.py`)
   - A run is confirmed by an explicit size label (`12"ø`, `22"x14"`, or `12"0` where OCR read the ø as a zero). The label must sit inside the run or beside it, lie within its length, and agree with its measured width. A label that two runs fit about equally well is not used.
   - A bare number such as `08` (the marks lost to OCR) counts only inside a run no explicit label claims, and only on or next to ductwork already confirmed. Callouts, CFM values and room numbers elsewhere are ignored.
   - Connected runs of the same width inherit that size (`inferred`).
   - Connected runs of a different width are kept with their measured width (`measured`) only if they pass all of these:
     - the same lineweight as confirmed ducts;
     - long enough;
     - not part of a flex-rib stack;
     - not the gap between, or a piece inside, accepted ducts.
   - Everything else is dropped. This is what keeps table borders, walls and equipment outlines out of the result.
6. **Supply/return** (`classify.py`)
   - Boxes with corner-to-corner diagonals are air-device and riser symbols: an X means supply, a single diagonal means return.
   - Symbols that touch a run end (risers, inline boxes) seed that run's system. Each run takes the system of the nearest seed in connection hops.
   - A duct ends at a riser or device box, so two runs whose link passes through one are not joined.
   - Devices hung off flex sit too far from their branch to attach reliably. A run that no seed reaches, or that is equally near a supply and a return seed, stays `unclassified`.
7. **Length**: the straight centerline length of each run at the drawing scale. Elbows, flex and fittings are not included.

## Results on the sample

`samples/testset2.pdf` (sheet M2.0) takes about 15 s on a 20-thread laptop CPU, most of it OCR; it is slower when the cores are busy.

- **Scale:** inferred as 1/4"=1'-0", matching the title block.
- **Runs:** 20 in total:
  - 13 confirmed by their own label;
  - 5 that inherited a size from a connected run;
  - 2 kept with a measured width: the 20" kitchen trunk and a 6" connector, neither of which carries a label.
- **Size labels:** all 13 labels on detected runs are read and matched correctly. Labels on elbows, flex and short collars are not used, because those pieces are not measured.
- **Systems:** 9 supply, 5 return, 6 unclassified (the unclassified runs are listed under Limitations). The 18"ø grease duct (18'-10") is kitchen exhaust. It stays unclassified because no supply or return symbol touches it; the tool has no exhaust class.
- **Nothing is marked** in the title block, the notes or the architectural background.
- **Missed runs**, compared with the reference annotation:
  - The 10"ø drop under the RTU-1 supply riser, about 4'-8". The 12"ø crossover splits its walls into pieces too short to pair, so its label (read correctly) has no run to attach to.
  - The 4"ø restroom exhaust. OCR merges its label with the adjacent "BDD" text, so the run is never confirmed.
  - The short 14"ø and 12"ø collars at tees.

## Tested on other drawings

To check the tool beyond the sample, I ran it on sheets from publicly posted drawing sets. These PDFs are not included in this repository.

| Drawing | Sheet | Style | Result |
|---|---|---|---|
| [Eglin AFB NICoE](https://imlive.s3.amazonaws.com/Federal%20Government/ID66990768310963037007432025819233376360/Attachment%204%20-%20Drawings%2006%20Mechanical.pdf) | M2.3 ground floor duct plan (page 3) | real text layer, 1/8"=1'-0" | Scale inferred as 1/8". 235 runs, 61 confirmed by their own label; spot-checked sizes match the drawing. The 162 measured runs were not checked one by one. |
| [CAD Sultants HVAC shop drawing](https://caddsultants.com/wp-content/uploads/2020/08/mechanical2.pdf) | M-101 (page 0) | ducts drawn in blue, real text layer, 3/8"=1'-0" | Refused by default, because the blue walls are lighter than the cut-off. With `--max-luma 0.35`: scale 3/8" from the sheet note, 63 runs, 59 confirmed by label. |
| [USC Lieber College renovation](https://sc.edu/purchasing/solicitations/documents/s_1509557045.pdf) | M-1-R to M-4-R (pages 15–21) | SHX text, labels on leader lines, dark ceiling grid, 1/4"=1'-0" | Three sheets refuse to infer the scale; M-4-R infers 1/4". Given the scale, the output is dominated by ceiling-grid cells and misses most ducts. This drawing style is not supported (see Limitations). |
| [White Sturgeon hatchery](https://static1.squarespace.com/static/56a24f7f841aba12ab7ecfa9/t/668c4aba6c8d156ad224713d/1720470211832/CCT+White+Sturgeon+CONSOLIDATED_Part2.pdf) | M-101 (page 0) | no duct size labels | Refuses: too few labels. |
| [Dundas Public School](https://www.schoolinfrastructure.nsw.gov.au/content/dam/infrastructure/projects/d/dundas-public-school-upgrade/2025/may/DPS_REF_-_A9_Mechanical_Drawings.PDF) | M-120 (page 3) | metric (mm) sizes | Refuses: too few labels. |

## Limitations

- **Vector PDFs only.** Scanned or rasterised drawings would need a raster wall detector; this tool has none.
- **Straight runs only.** Lengths exclude elbows, flex and fittings. A straight stub shorter than about twice its width is not detected, so on the sample some short collars at tees are missing.
- **Colour assumption.** Ductwork is assumed dark on a lighter background. Sheets that draw ducts in colour or grey need `--max-luma` raised.
- **Label placement.** A label must sit inside its duct, or within about 12 pt of it with an explicit size mark, and within the duct's length. A label equally close to two matching ducts is left unused.
- **Leader-line drawings.** Leader lines are not followed. Drawings that put their size labels at the end of leader lines, like the USC set above, leave most ducts unconfirmed. When such a drawing also draws its ceiling grid dark, grid cells are grown as measured ducts.
- **Imperial sizes only.** Metric labels (`600x300`, `Ø90`) are not parsed.
- **Rectangular ducts.** For `22"x14"` only the plan dimension can be checked against geometry; the depth comes from the label alone.
- **Supply/return is a heuristic.** It depends on symbol conventions, not the air-device schedule.
  - On the sample, six runs are unclassified:
    - the grease duct;
    - the upper 14" dining duct and the 10" branch from grille B/375, whose stubs to the DOAS-1 box are too short to be detected as runs;
    - the 12"ø drop beside the RTU-1 risers and its branches to A/700 and D/500, which reach the return riser only through an elbow, not a detected run.
  - Reliable classification needs the schedule sheet or layers that encode the system.
- **Thresholds.** With an inferred scale, wall pairing uses the 4–80 pt paper defaults: about 2.7"–53" at 1/4"=1'-0", but 5.3"–107" at 1/8". Pass `--scale` on sheets at other scales, so duct widths and symbol sizes follow the real sizes. The remaining thresholds (label offset, touch tolerance, OCR glyph sizes) are paper-space conventions. All were tuned on this one sheet.
- **Memory.** Peak memory is about 1.2 GB, from the full-sheet 300 dpi render and its component maps. Rendering only the regions around candidate runs would cut it.
- **Other gaps:** dashed (hidden or existing) ductwork is not detected, and one page is processed per run.

## Tests

```bash
uv run pytest               # all tests, ~25 s on 20 threads
uv run pytest -m "not slow" # skip the full OCR run on the sample
```

- Unit tests cover:
  - wall pairing on synthetic geometry: transitions, adjacent ducts, flex ribs, liners, diagonals;
  - run merging and connectivity;
  - label and scale parsing, including real OCR strings from the sample;
  - label assignment, including ambiguous labels, labels past a run's end and bare numbers;
  - scale inference: clear, tied and narrow votes, and the sheet's noted scale;
  - run growth;
  - symbol classification, including conflicting seeds and links through riser boxes.
- `tests/test_pipeline.py` runs synthetic drawings with a real text layer through detection and every output, without OCR. This includes drawings at 1/2" and 1/8" scale whose ducts fall outside the paper-point defaults.
- `tests/test_sample.py` checks the known runs on the sample drawing, and pins the full takeoff: run counts by source and by system.

## Dependencies

- PyMuPDF: vector extraction, rendering and PDF output. It is licensed AGPL-3.0; commercial use needs a license from Artifex.
- NumPy and OpenCV: geometry and image processing.
- `rapidocr-onnxruntime`: PP-OCRv4 models on ONNX Runtime, CPU only. The models ship inside the package, so no download is needed at run time.
