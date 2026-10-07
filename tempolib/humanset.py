"""Turn the annotated anchors into dataset rows."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from . import extraction_fr
from .cleaning import clean_chunk, ensure_contains
from .dedup import normalize
from .intervals import anchor_interval
from .schema import new_row
from .textclean import clean_claim, window_document

_MODERN_YEAR = re.compile(r"(?:19|20)\d{2}")
_GRADED = {"generalization", "precision", "reformulation"}
_VALIDITY_LABEL = {"valid": 1, "invalid": 0}


def is_citation_year(anchor: dict) -> bool:
    expr = str(anchor.get("expression", "")).strip()
    if not _MODERN_YEAR.fullmatch(expr):
        return False
    sentence = str(anchor.get("sentence", ""))
    e = re.escape(expr)
    if re.search(rf"\[[^\]]*{e}[^\]]*\]", sentence) or re.search(rf"{e}\s*\]", sentence):
        return True
    if re.search(rf"\[[^\]]{{0,80}}{e}", sentence):
        return True
    return str(anchor.get("source_pub_year") or "").strip()[:4] == expr


def build_rows(
    anchors: Iterable[dict],
    split: str,
    strip_bib: bool = True,
    tempolib_version: str = "",
    keep_positive: bool = True,
    keep_negative: bool = True,
    doc_window: int = 0,
) -> tuple[list[dict], dict]:
    """Turn annotated anchors into dataset rows: the sentence is the claim, the source chunk the document. Rows with the same claim and document are merged, and an invalid label wins."""
    stats: Counter = Counter()
    merged: dict[tuple, dict] = {}

    for a in anchors:
        et = (a.get("error_type") or "none").strip() or "none"

        if is_citation_year(a):
            stats["skipped (citation year)"] += 1
            continue

        if et == "none":
            label = 1
        elif et == "wrong_date":
            label = 0
        elif et in _GRADED:
            v = (a.get("validity") or "").strip().lower()
            if v == "reject":
                stats["rejected (validity)"] += 1
                continue
            if v not in _VALIDITY_LABEL:
                stats["generalization/precision without validity -> skipped"] += 1
                continue
            label = _VALIDITY_LABEL[v]
        else:
            stats[f"unknown type: {et}"] += 1
            continue

        if label == 0 and not keep_negative:
            stats["negatives skipped (option)"] += 1
            continue
        if label == 1 and not keep_positive:
            stats["positives skipped (option)"] += 1
            continue

        source_window = a.get("source_window", "") or ""
        document = ensure_contains(
            clean_chunk(a.get("source_chunk", ""), strip_bibliography=strip_bib),
            source_window,
        )
        if doc_window:
            focus = source_window if source_window not in ("", "(no match)") else (a.get("sentence") or "")
            document = window_document(document, focus, max_tokens=doc_window)
        raw = (a.get("sentence") or "").strip()
        expr = str(a.get("expression", "")).strip()
        claim = clean_claim(raw)
        if expr and expr not in claim:
            claim = raw
        if not document.strip() or not claim:
            stats["skipped (empty)"] += 1
            continue
        if claim != raw:
            stats["claims cleaned"] += 1

        anc = anchor_interval(expr, raw, extraction_fr)
        row = new_row(
            document=document,
            claim=claim,
            label=label,
            error_type=et,
            granularity=anc[1] if anc else None,
            anchor_expression=expr,
            original_sentence=raw,
            doc_id=a.get("doc_id", "doc_?"),
            source_pub_year=a.get("source_pub_year"),
            stratum=a.get("stratum"),
            split=split,
            origin="human",
            verification={
                "origin": "human_annotation",
                "annotation_raw": a.get("annotation"),
                "suggested_relation": a.get("suggested_relation"),
                "label_rule": None,
                "label_llm": None,
            },
            provenance={
                "llm_model": None, "llm_seed": None, "prompt_version": None,
                "tempolib_version": tempolib_version, "generated_at": None,
            },
        )

        k = (normalize(claim), row["doc_id"])
        if k in merged:
            stats["merged"] += 1
            if row["label"] == 0 and merged[k]["label"] == 1:
                merged[k].update(label=0, error_type=row["error_type"],
                                 granularity=row["granularity"],
                                 anchor_expression=row["anchor_expression"])
        else:
            merged[k] = row

    rows = list(merged.values())
    stats["rows"] = len(rows)
    stats["pos"] = sum(1 for r in rows if r["label"] == 1)
    stats["neg"] = sum(1 for r in rows if r["label"] == 0)
    return rows, dict(stats)
