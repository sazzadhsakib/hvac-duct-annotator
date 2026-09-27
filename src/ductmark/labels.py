import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DuctSize:
    width: int
    height: int | None = None

    @property
    def shape(self) -> str:
        return "round" if self.height is None else "rectangular"

    def __str__(self) -> str:
        return f'{self.width}"ø' if self.height is None else f'{self.width}"x{self.height}"'


MIN_SIZE, MAX_SIZE = 3, 60

# OCR renders the SHX "ø" as 0, o, g, Φ, a CJK glyph, or drops it; inch marks come back as
# quotes, doubled apostrophes, asterisks or nothing. A single apostrophe is feet, not inches.
_CLEAN = str.maketrans({"I": "1", "l": "1", "|": None, "[": None, "]": None, "(": None, ")": None, " ": None})
_INCH = "(?:[\"”″*]|'')"
_DIA = "øØΦφ⌀∅g@中"
_RECT = re.compile(rf"(\d{{1,2}}){_INCH}?[xX×](\d{{1,2}}){_INCH}?")
_QUOTED = re.compile(rf"(\d{{1,2}}){_INCH}{{1,2}}([{_DIA}0oO])?")
_DIAMETER = re.compile(rf"(\d{{1,2}})[{_DIA}]")
# An inch mark alone is not enough: dimensions and heights ("8'-6\"") carry those too.
_MARK = re.compile(rf"[{_DIA}]|{_INCH}[0oO]$|\d{_INCH}?[xX×]\d")


def size_readings(text: str) -> list[DuctSize]:
    """Every plausible duct size a raw OCR string could stand for, most literal first."""
    t = text.strip().translate(_CLEAN)
    if m := _RECT.fullmatch(t):
        sizes = [DuctSize(int(m[1]), int(m[2]))]
    elif m := _QUOTED.fullmatch(t) or _DIAMETER.fullmatch(t):
        sizes = [DuctSize(int(m[1]))]
    elif t.isascii() and t.isdigit() and len(t) <= 4:
        # Bare digits: the inch mark and/or ø were read as digits ("80", "140", "8110"),
        # so the real size is a prefix. The measured width decides later.
        prefixes = [t] if len(t) <= 2 else []
        sizes = [DuctSize(int(p)) for p in dict.fromkeys(prefixes + [t[:2], t[:1]])]
    else:
        return []
    return [s for s in sizes if MIN_SIZE <= s.width <= MAX_SIZE and (s.height is None or MIN_SIZE <= s.height <= MAX_SIZE)]


def has_size_mark(text: str) -> bool:
    return bool(_MARK.search(text))


def pick(readings: list[DuctSize], measured_in: float, rel_tol: float = 0.08, abs_tol: float = 1.0) -> DuctSize | None:
    """The reading whose plan-visible dimension agrees with the measured wall gap."""
    best, best_err = None, None
    for size in readings:
        for dim in (size.width, size.height):
            if dim is None:
                continue
            err = abs(dim - measured_in)
            if err <= max(abs_tol, rel_tol * dim) and (best_err is None or err < best_err):
                best, best_err = size, err
    return best


def parse_scale(text: str) -> float:
    """Drawing scale as paper points per real inch, e.g. 1/4"=1'-0" -> 1.5."""
    t = text.replace(" ", "").replace("”", '"').replace("’", "'")
    if m := re.fullmatch(r"(?:(\d+)-)?(\d+)(?:/(\d+))?\"=1'(?:-0\"?)?", t):
        whole, num, den = m.groups()
        paper_in_per_ft = int(whole or 0) + int(num) / int(den or 1)
        return paper_in_per_ft * 72 / 12
    if m := re.fullmatch(r"1\"=(\d+)'(?:-0\"?)?", t):
        return 72 / (int(m[1]) * 12)
    if m := re.fullmatch(r"1:(\d+)", t):
        return 72 / int(m[1])
    raise ValueError(f"unrecognised scale: {text!r}")


STANDARD_SCALES = {
    s: parse_scale(s)
    for s in (
        "3/32\"=1'-0\"", "1/8\"=1'-0\"", "3/16\"=1'-0\"", "1/4\"=1'-0\"", "3/8\"=1'-0\"", "1/2\"=1'-0\"",
        "3/4\"=1'-0\"", "1\"=1'-0\"", "1-1/2\"=1'-0\"", "3\"=1'-0\"",
        "1\"=10'", "1\"=20'", "1\"=30'", "1\"=40'", "1\"=50'", "1\"=100'",
    )
}


def infer_scale(observations: list[tuple[list[DuctSize], float]], min_votes: int = 2) -> tuple[str, float]:
    """Pick the standard scale under which the most labels agree with their measured wall gaps.

    observations: (size readings of a label, wall gap in points of the run it sits in).
    """
    votes = {
        name: sum(pick(readings, gap_pt / ppi) is not None for readings, gap_pt in observations)
        for name, ppi in STANDARD_SCALES.items()
    }
    name = max(votes, key=votes.get)
    if votes[name] < min_votes:
        raise ValueError("could not infer drawing scale from duct labels; pass --scale")
    return name, STANDARD_SCALES[name]


def format_ft_in(feet: float) -> str:
    inches = round(feet * 12)
    return f"{inches // 12}'-{inches % 12}\""
