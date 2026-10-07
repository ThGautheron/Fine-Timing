"""Language-independent building blocks: expression records and interval relations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MIN_YR, MAX_YR = 100, 2030
THRESHOLD_REGISTRY = 1800


@dataclass
class Normalization:
    """An expression turned into an interval of years, with its granularity."""

    start: Optional[int] = None
    end: Optional[int] = None
    granularity: Optional[str] = None
    is_valid: bool = True


@dataclass
class Expression:
    """A temporal expression found in a text, with its position and its normalised interval."""

    surface: str
    char_start: int
    char_end: int
    expr_type: str
    normalization: Normalization = field(default_factory=Normalization)


def century_to_interval(n: int) -> tuple[int, int]:
    return (n - 1) * 100 + 1, n * 100


def get_registry(start: Optional[int], end: Optional[int]) -> Optional[str]:
    ref = start if start is not None else end
    if ref is None:
        return None
    return "ancient" if ref < THRESHOLD_REGISTRY else "modern"


def compatible_registries(s_start: Optional[int], s_end: Optional[int],
                          c_start: Optional[int], c_end: Optional[int]) -> bool:
    rs, rc = get_registry(s_start, s_end), get_registry(c_start, c_end)
    if not rs or not rc:
        return True
    if s_start is not None and s_end is not None and s_start < THRESHOLD_REGISTRY <= s_end:
        return True
    if c_start is not None and c_end is not None and c_start < THRESHOLD_REGISTRY <= c_end:
        return True
    return rs == rc


def compare_intervals(s_start: Optional[int], s_end: Optional[int],
                      c_start: Optional[int], c_end: Optional[int]) -> Optional[str]:
    """How the second interval relates to the first: equal, included, contains, disjoint or overlaps."""
    if (s_start is None and s_end is None) or (c_start is None and c_end is None):
        return None
    if not compatible_registries(s_start, s_end, c_start, c_end):
        return None

    sd = float("-inf") if s_start is None else float(s_start)
    se = float("inf") if s_end is None else float(s_end)
    cd = float("-inf") if c_start is None else float(c_start)
    ce = float("inf") if c_end is None else float(c_end)

    if sd > se or cd > ce:
        return None
    if sd == cd and se == ce:
        return "equal"
    if se < cd or sd > ce:
        return "disjoint"
    if sd >= cd and se <= ce:
        return "included"
    if sd <= cd and se >= ce:
        return "contains"
    return "overlaps"


def get_window(text: str, expr: Expression, window: int = 40) -> str:
    return text[max(0, expr.char_start - window): min(len(text), expr.char_end + window)]


def clean_md(sentence: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^[#>\s\-]+|\*{1,3}", "", sentence)).strip()
