"""Counts behind the data description of the paper: the anchors (Sect. 4.1), the rewrites and their labels (Sect. 4.2), and Tables 2 and 4.

Run it after the `splits` and `datasets` stages. The numbers are written to `results/statistics/data.json`. The number of
documents per journal (Table 2) is not computed, since the journals are not distributed."""
from __future__ import annotations

from collections import Counter

from tempolib import read_jsonl, resolve, setup_console
from tempolib.evaluation import CORPORA, STATISTICS_DIR, anchor_table, load_expert_pairs, rag_records, write_json
from tempolib.humanset import is_citation_year

TYPES = ["generalization", "precision", "reformulation", "wrong_date"]
TYPE_OF_ANNOTATION = {"none": "reformulation"}


def type_shares(types: list[str]) -> dict:
    c = Counter(TYPE_OF_ANNOTATION.get(t, t) for t in types)
    return {"n": len(types), **{t: round(100 * c[t] / len(types), 1) for t in TYPES}}


def rows_of(path: str) -> list[dict]:
    return list(read_jsonl(resolve(path)))


def main() -> int:
    setup_console()
    out: dict = {}

    anchors = {c: anchor_table(c) for c in CORPORA}
    typed = {c: [r for r in rows if (r["annotation"] or "").strip()] for c, rows in anchors.items()}
    all_anchors = [r for rows in anchors.values() for r in rows]
    all_typed = [r for rows in typed.values() for r in rows]
    undecided = [r for r in all_anchors if not (r["annotation"] or "").strip()]
    out["anchors"] = {
        "queries": {c: len(rag_records(c)) for c in CORPORA},
        "expressions": len(all_anchors),
        "typed": len(all_typed), "typed_share": round(100 * len(all_typed) / len(all_anchors), 1),
        "undecided": len(undecided),
        "undecided_disjoint": sum(r["suggested_relation"] == "disjoint" for r in undecided),
    }
    out["table4_rag_summaries"] = type_shares([r["annotation"].strip().lower() for r in all_typed])
    for r in out["anchors"], out["table4_rag_summaries"]:
        print(r)

    for lang in ("en", "fr"):
        kept = 0
        for s in ("train", "dev"):
            for r in rows_of(f"outputs/splits/{lang}/anchors_{s}.jsonl"):
                faithful = r["error_type"] == "none" or (r["error_type"] == "generalization" and r.get("validity") == "valid")
                kept += faithful and not is_citation_year(r)
        out["anchors"][f"{lang}_train_dev_reformulations_and_valid_generalizations"] = kept
        print(f"[{lang}] reformulations and valid generalizations in train+dev (citation years excluded): {kept}")

    candidates = rows_of("outputs/work/en/judged.jsonl")
    judged = [c for c in candidates if c["agrees_with_polarity"] is not None]
    asked_valid = [c for c in judged if c["polarity"] == "valid"]
    out["rewrites"] = {
        "candidates": len(candidates), "judged": len(judged),
        "agreement_requested_label_vs_criterion": round(100 * sum(c["agrees_with_polarity"] for c in judged) / len(judged), 1),
        "valid_under_criterion_among_requested_valid": round(
            100 * sum(c["rule_verdict"] == "valid" for c in asked_valid) / len(asked_valid), 1)}
    print(out["rewrites"])

    final = {"en": {s: rows_of(f"outputs/datasets/en_main/{s}.jsonl") for s in ("train", "dev")},
             "fr": {s: rows_of(f"outputs/datasets/fr/{s}.jsonl") for s in ("train", "dev")}}
    table2 = {}
    for lang, parts in final.items():
        rows = parts["train"] + parts["dev"]
        groups = {c: [r for r in rows if r["doc_id"].startswith(f"{c}:")] for c in (CORPORA if lang == "en" else ("fr",))}
        for c, g in groups.items():
            table2[c] = {"train": sum(r["split"] == "train" for r in g), "dev": sum(r["split"] == "dev" for r in g),
                         "valid": sum(r["label"] == 1 for r in g), "invalid": sum(r["label"] == 0 for r in g)}
        if lang == "en":
            table2["en_total"] = {k: sum(table2[c][k] for c in CORPORA) for k in ("train", "dev", "valid", "invalid")}
        out[f"table4_rewrites_{lang}"] = type_shares([r["error_type"] for r in rows])
        out[f"origins_{lang}"] = dict(Counter(r["origin"] for r in rows))
        print(lang, out[f"table4_rewrites_{lang}"], out[f"origins_{lang}"])
    out["table2"] = table2
    for k, v in table2.items():
        print(f"  {k:9}", v)
    silver = [r for r in final["en"]["train"] + final["en"]["dev"] if r["origin"] == "silver"]
    out["silver"] = {"n": len(silver), "train": sum(r["split"] == "train" for r in silver),
                     "dev": sum(r["split"] == "dev" for r in silver),
                     "types": {t: round(100 * n / len(silver), 1) for t, n in Counter(r["error_type"] for r in silver).items()}}
    print(out["silver"])

    pairs, _ = load_expert_pairs("en")
    binary = [r for r in pairs if not (r["error_type"] == "reformulation" and r["label"] == 0)]
    out["table4_expert"] = type_shares([r["error_type"] for r in binary])
    out["expert_decided_pairs"] = {"n": len(pairs), "valid": sum(r["label"] for r in pairs)}
    print(out["table4_expert"], out["expert_decided_pairs"])
    write_json(STATISTICS_DIR / "data.json", out, indent=1)
    print("\n-> results/statistics/data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
