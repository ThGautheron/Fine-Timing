"""Granularity levels of date intervals, and the interval of an anchor expression."""
from __future__ import annotations

from types import ModuleType
from typing import Optional

from . import extraction_fr

LEVELS = [
    (1, 0, "day"),
    (100, 1, "month-season"),
    (400, 2, "year"),
    (4000, 3, "decade"),
    (20000, 4, "half-century"),
    (50000, 5, "century"),
    (150000, 6, "period-narrow"),
    (400000, 7, "period-wide"),
    (float("inf"), 8, "period-era"),
]
DAYS_PER_YEAR = 365
_FINE = {"day": 0, "month": 1, "season": 1}


def granularity_level(interval: Optional[tuple], granularity: Optional[str] = None) -> Optional[int]:
    """0 (a day) to 8 (an era), from the stated granularity or else from the width of the interval."""
    if granularity in _FINE:
        return _FINE[granularity]
    if not interval or interval[0] is None or interval[1] is None:
        return None
    width = (interval[1] - interval[0] + 1) * DAYS_PER_YEAR
    return next((level for max_width, level, _ in LEVELS if width <= max_width), 8)


def anchor_interval(expression: str, sentence: str = "", backend: ModuleType = extraction_fr) -> Optional[tuple]:
    return backend.first_interval(expression) or (backend.first_interval(sentence) if sentence else None)
