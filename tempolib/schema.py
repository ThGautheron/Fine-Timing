"""What a dataset row looks like: document, claim, label and provenance."""
from __future__ import annotations

import hashlib

ERROR_TYPES = {"none", "generalization", "precision", "reformulation", "wrong_date"}
GRANULARITIES = {
    None, "day", "month", "season", "year", "decade", "century", "period", "unknown",
}
SPLITS = {"train", "dev", "test"}
ORIGINS = {"synthetic", "human", "silver"}

REQUIRED = [
    "id", "document", "claim", "label", "error_type", "granularity",
    "anchor_expression", "anchor_expression_source", "anchor_expression_claim",
    "original_sentence", "doc_id", "source_pub_year", "stratum",
    "split", "origin", "verification", "provenance",
]

PROVENANCE_KEYS = {"llm_model", "llm_seed", "prompt_version", "tempolib_version", "generated_at"}


def make_id(doc_id: str, anchor_expression: str, error_type: str, claim: str) -> str:
    raw = f"{doc_id}||{anchor_expression}||{error_type}||{claim}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def new_row(**kw) -> dict:
    """A dataset row with every required field (missing ones are None) and its id."""
    row = {k: kw.get(k) for k in REQUIRED}
    row["id"] = make_id(
        row["doc_id"] or "",
        row["anchor_expression"] or "",
        row["error_type"] or "",
        row["claim"] or "",
    )
    return row


def validate(row: dict) -> list[str]:
    """The problems of a row, as readable messages; an empty list means the row is valid."""
    errs: list[str] = []
    for k in REQUIRED:
        if k not in row:
            errs.append(f"missing field: {k}")
    if errs:
        return errs

    if row["label"] not in (0, 1):
        errs.append(f"invalid label: {row['label']!r}")
    if row["error_type"] not in ERROR_TYPES:
        errs.append(f"invalid error_type: {row['error_type']!r}")
    if row["granularity"] not in GRANULARITIES:
        errs.append(f"invalid granularity: {row['granularity']!r}")
    if row["split"] not in SPLITS:
        errs.append(f"invalid split: {row['split']!r}")
    if row["origin"] not in ORIGINS:
        errs.append(f"invalid origin: {row['origin']!r}")
    if not str(row["document"]).strip():
        errs.append("empty document")
    if not str(row["claim"]).strip():
        errs.append("empty claim")
    if row["source_pub_year"] is not None and not (100 <= int(row["source_pub_year"]) <= 2100):
        errs.append(f"source_pub_year out of range: {row['source_pub_year']!r}")
    if not isinstance(row["verification"], dict):
        errs.append("verification missing or not a dict")
    prov = row["provenance"]
    if not isinstance(prov, dict) or not PROVENANCE_KEYS.issubset(prov):
        errs.append("incomplete provenance")
    return errs
