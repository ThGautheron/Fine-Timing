"""Read the annotated anchor tables: one row per dated expression of a summary sentence."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .utils import open_text, resolve

ERROR_TYPE_NORM = {
    "": "none",
    "none": "none",
    "generalization": "generalization",
    "precision": "precision",
    "reformulation": "reformulation",
    "wrong_date": "wrong_date",
    "invented_date": "wrong_date",
}

STRATUM_NORM = {
    "": None, "none": None, "object": "object", "source": "source", "writing": "writing",
    "undecidable": "undecidable",
}


@dataclass
class AnchorRow:

    item_id: str
    doc_id: str
    sentence: str
    expression: str
    source_chunk: str
    source_window: str = ""
    source_file: Optional[str] = None
    source_pub_year: Optional[int] = None
    stratum: Optional[str] = None
    annotation: str = "none"
    suggested_relation: Optional[str] = None
    validity: Optional[str] = None
    corpus: str = ""

    @property
    def error_type(self) -> str:
        key = str(self.annotation).strip().lower()
        if key not in ERROR_TYPE_NORM:
            raise ValueError(f"unknown annotation {self.annotation!r} (item {self.item_id})")
        return ERROR_TYPE_NORM[key]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["error_type"] = self.error_type
        return d


def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    s = str(value).strip()
    return bool(s) and s.lower() != "nan"


def _pub_year(value) -> Optional[int]:
    if not _present(value):
        return None
    s = str(value).strip()
    return int(s[:4]) if s[:4].isdigit() else None


def _clean_opt(value) -> Optional[str]:
    return str(value).strip() if _present(value) else None


def _from_record(rec: dict, corpus: str) -> Optional[AnchorRow]:
    """An anchor row from a raw table record, or None if the sentence, the expression or the chunk is missing."""
    def g(*keys):
        for k in keys:
            if k in rec and _present(rec[k]):
                return rec[k]
        return None

    chunk = g("source_chunk", "source_window")
    expr = _clean_opt(g("expression"))
    sentence = _clean_opt(g("sentence"))
    if not chunk or not expr or not sentence:
        return None

    stratum_raw = (_clean_opt(g("stratum")) or "").lower()
    return AnchorRow(
        item_id=str(g("item_id") or ""),
        doc_id=str(g("doc_id") or "doc_?").strip(),
        sentence=sentence,
        expression=expr,
        source_chunk=str(chunk).strip(),
        source_window=_clean_opt(g("source_window")) or "",
        source_file=_clean_opt(g("source_file")),
        source_pub_year=_pub_year(g("source_pub_year")),
        stratum=STRATUM_NORM.get(stratum_raw, stratum_raw or None),
        annotation=str(g("annotation") or "none"),
        suggested_relation=_clean_opt(g("suggested_relation")),
        validity=(_clean_opt(g("validity")) or "").lower() or None,
        corpus=corpus,
    )


def _load_jsonl(path: Path) -> list[dict]:
    with open_text(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_csv(path: Path) -> list[dict]:
    with open_text(path, newline="") as f:
        return list(csv.DictReader(f))


_READERS = {"jsonl": _load_jsonl, "csv": _load_csv}


def read_anchor_table(path: str | Path, annotated_only: bool = False) -> list[dict]:
    path = Path(path)
    fmt = path.name.removesuffix(".gz").rsplit(".", 1)[-1].lower()
    if fmt not in _READERS:
        raise ValueError(f"unsupported anchor format: {fmt!r} ({path})")
    records = _READERS[fmt](path)
    if annotated_only:
        records = [r for r in records if str(r.get("annotation") or "").strip()]
    return records


def load_corpus(path: str | Path, name: str, annotated_only: bool = False) -> list[AnchorRow]:
    rows = []
    for i, rec in enumerate(read_anchor_table(path)):
        if annotated_only and not str(rec.get("annotation") or "").strip():
            continue
        ar = _from_record(rec, name)
        if ar is None:
            continue
        ar.item_id = f"{name}:{ar.item_id}" if ar.item_id else f"{name}:{i}"
        ar.doc_id = f"{name}:{ar.doc_id}"
        rows.append(ar)
    return rows


def load_all(specs: list[dict]) -> list[AnchorRow]:
    out: list[AnchorRow] = []
    for spec in specs:
        out.extend(load_corpus(resolve(spec["path"]), spec["name"], bool(spec.get("annotated_only", False))))
    return out
