"""Turn judged rewrites into labelled ones, using either the verdict of the criterion or the label the generator was asked for."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tempolib import read_jsonl, resolve, setup_console, write_jsonl
from tempolib.dedup import normalize


def drop_conflicts(rows: list[dict], stats: Counter) -> list[dict]:
    labels: dict = {}
    for c in rows:
        labels.setdefault((c.get("doc_id", ""), normalize(c["claim"])), set()).add(c["verification"]["label"])
    bad = {k for k, v in labels.items() if len(v) > 1}
    kept = [c for c in rows if (c.get("doc_id", ""), normalize(c["claim"])) not in bad]
    stats["excluded: contradictory labels (rewrites)"] = len(rows) - len(kept)
    stats["excluded: contradictory labels (distinct claims)"] = len(bad)
    return kept


def convert(rows: list[dict], label_source: str = "rule", keep_all: bool = False) -> tuple[list[dict], Counter]:
    """Attach a label to each usable rewrite, from the criterion or from the generator, and count what is left out."""
    out = []
    stats = Counter()
    for r in rows:
        stats["total"] += 1
        if r["claim"].strip() == r["original_sentence"].strip():
            stats["excluded: null transformation"] += 1
            continue
        verdict = r.get("rule_verdict")
        if not keep_all:
            if verdict is None:
                stats[f"excluded: {r.get('rule_class') or 'not judged'}"] += 1
                continue
            if verdict == "tolerated":
                stats["excluded: tolerated"] += 1
                continue
        rule_label = None if verdict in (None, "tolerated") else verdict == "valid"
        if label_source == "llm":
            if r.get("label_llm") is None:
                stats["excluded: generator label missing"] += 1
                continue
            label = bool(r["label_llm"])
            if rule_label is not None:
                stats["generator != rule"] += label != rule_label
        else:
            label = rule_label
        stats[f"kept: {verdict or 'not judged'}"] += 1
        c = dict(r)
        c["verification"] = {
            "label_llm": r.get("label_llm"),
            "label_rule": rule_label,
            "label": label,
            "label_source": label_source,
            "rule_reason": r.get("rule_reason"),
            "interval_anchor": r.get("interval_anchor"),
        }
        out.append(c)
    return out, stats


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("judged", type=Path, help="output of pipeline.judge_rewrites")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--label-source", choices=["rule", "llm"], default="rule",
                    help="rule verdict (default) or the polarity the generator was asked for")
    ap.add_argument("--drop-conflicts", action="store_true",
                    help="drop rewrites present with both labels (the labeller contradicts itself)")
    ap.add_argument("--keep-all", action="store_true",
                    help="with --label-source llm: keep every rewrite, including those the rule cannot judge")
    a = ap.parse_args()
    if a.keep_all and a.label_source != "llm":
        ap.error("--keep-all only makes sense with --label-source llm")
    rows = list(read_jsonl(resolve(a.judged)))
    out, stats = convert(rows, a.label_source, a.keep_all)
    if a.drop_conflicts:
        out = drop_conflicts(out, stats)
    n = write_jsonl(resolve(a.output), out)
    print(f"{a.judged}: {len(rows)} rows -> {n} labelled (labels: {a.label_source}) -> {a.output}")
    for k, v in sorted(stats.items()):
        print(f"  {k:48s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
