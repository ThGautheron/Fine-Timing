"""Paths and helpers for the expert set and for the stored model probabilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .anchors import read_anchor_table
from .utils import file_sha, read_jsonl, resolve

CLASSES = ["generalization", "precision", "reformulation", "wrong_date"]
EXPERT_SET = resolve("data/expert_set.json")
REFERENCE_DIR = resolve("data/reference")
RESULTS_DIR = resolve("results")
PROBABILITIES_DIR = RESULTS_DIR / "probabilities"
TABLES_DIR = RESULTS_DIR / "tables"
STATISTICS_DIR = RESULTS_DIR / "statistics"
RETENTION_DIR = RESULTS_DIR / "retention"
JUDGE_DIR = RESULTS_DIR / "llm_judge"
THRESHOLD = 0.5
CORPORA = ("dhq", "dm", "chdj")


def expert_sha() -> str:
    return file_sha(EXPERT_SET)


def load_expert_pairs(lang: str = "en") -> tuple[list[dict], int]:
    data = json.loads(EXPERT_SET.read_text(encoding="utf-8"))
    rows, undecided = [], 0
    for it in data["items"]:
        for cls in CLASSES:
            for v in it["variants"].get(cls, []):
                if v["valid"] is None:
                    undecided += 1
                    continue
                rows.append({"id": it["id"], "document": it[f"doc_{lang}"], "claim": v[f"text_{lang}"],
                             "error_type": cls, "label": int(bool(v["valid"]))})
    return rows, undecided


def evaluation_sets() -> dict[str, list[dict]]:
    pairs, _ = load_expert_pairs("en")
    return {"expert": [{"doc": r["document"], "claim": r["claim"], "cls": r["error_type"], "label": r["label"]}
                       for r in pairs]}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj, indent: Optional[int] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_results(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"missing {path}: run the script that produces it first")
    return read_json(path)


def probability_store(name: str) -> dict:
    ref = REFERENCE_DIR / name
    store = read_json(ref) if ref.exists() else {}
    store.setdefault("probs", {})
    return store


def anchor_table(corpus: str, annotated_only: bool = False) -> list[dict]:
    return read_anchor_table(resolve(f"data/anchors/{corpus}.csv.gz"), annotated_only)


def rag_records(corpus: str) -> list[dict]:
    return list(read_jsonl(resolve(f"data/rag_outputs/{corpus}.jsonl")))
