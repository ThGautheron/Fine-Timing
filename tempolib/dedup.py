"""Remove exact duplicate claims."""
from __future__ import annotations

import re
from typing import Callable

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def drop_exact(
    rows: list[dict],
    key: Callable[[dict], tuple] = lambda r: (normalize(r["claim"]), r.get("doc_id", "")),
) -> tuple[list[dict], int]:
    """Keep the first row of each (claim, document); also returns how many rows were dropped."""
    seen: set = set()
    kept: list[dict] = []
    for r in rows:
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        kept.append(r)
    return kept, len(rows) - len(kept)
