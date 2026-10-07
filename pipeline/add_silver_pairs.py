"""Add the silver rewrites to the training set (Sect. 4.2).

These are rewrites of simple dates that the rule could not decide and that were labelled by hand (`label_manual`).
They skip labelling, rebalancing and relabelling and are appended at the end."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from tempolib import __version__, read_jsonl, resolve, setup_console, write_jsonl
from tempolib.judge import EN, claim_expression, judge
from tempolib.schema import new_row, validate
from tempolib.tiers import parse

DEFAULT_SILVER = Path("data/rewrites/en_silver.jsonl.gz")


def build_rows(rewrites: list[dict], stats: Counter) -> list[dict]:
    """Dataset rows (origin `silver`) of the silver rewrites that the criterion can classify."""
    out = []
    for r in rewrites:
        verdict, cls, why = judge(r, EN)
        if verdict not in ("valid", "invalid"):
            stats["not judged"] += 1
            continue
        label = int(r["label_manual"])
        stats["manual != rule"] += label != int(verdict == "valid")
        expression = claim_expression(r["original_sentence"], r["claim"], EN)
        claimed = parse(expression)
        if expression.startswith("Between "):
            expression = "b" + expression[1:]
        row = new_row(
            document=r["document"], claim=r["claim"], label=label,
            error_type=cls, granularity=r["granularity"], anchor_expression=r["anchor_expression"],
            anchor_expression_source=r["anchor_expression"],
            anchor_expression_claim=expression,
            original_sentence=r["original_sentence"], doc_id=r["doc_id"],
            source_pub_year=r["source_pub_year"], stratum=r.get("stratum"),
            split=r["split"], origin="silver",
            verification={"label_llm": None, "label_rule": verdict == "valid", "label_manual": bool(label),
                          "rule_reason": why,
                          "interval_anchor": list(r["interval_anchor"]),
                          "interval_claim": [claimed["start"], claimed["end"]]},
            provenance={"generator": "pipeline.add_silver_pairs", "llm_model": r.get("llm_model"),
                        "llm_seed": r.get("llm_seed"), "prompt_version": r.get("prompt_version"),
                        "tempolib_version": __version__, "generated_at": r.get("generated_at")},
        )
        if validate(row):
            stats["schema rejected"] += 1
            continue
        out.append(row)
        stats[r["split"], cls, "valid" if label else "invalid"] += 1
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--silver", type=Path, default=DEFAULT_SILVER,
                    help="silver rewrites (JSON lines, optionally gzipped)")
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    in_dir, out_dir = resolve(args.in_dir), resolve(args.out_dir)

    stats = Counter()
    silver = build_rows(list(read_jsonl(resolve(args.silver))), stats)
    report = [f"add_silver_pairs: {args.silver} + {args.in_dir} -> {args.out_dir} (seed={args.seed})"]
    for split in ("train", "dev"):
        base = list(read_jsonl(in_dir / f"{split}.jsonl"))
        seen = {r["id"] for r in base}
        add = []
        for r in silver:
            if r["split"] == split and r["id"] not in seen:
                seen.add(r["id"])
                add.append(r)
        rows = base + add
        rng.shuffle(rows)
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        pos = sum(r["label"] for r in rows)
        report.append(f"[{split}] base {len(base)} + silver {len(add)} = {len(rows)} (pos={pos}/neg={len(rows) - pos})")
    report.append("\nrewrites per (split, class, verdict):")
    report += [f"  {k[0]:5} {k[1]:16} {k[2]:8} {v}" for k, v in sorted(stats.items(), key=str) if len(k) == 3]
    report.append(f"\nnot judged: {stats['not judged']} | schema rejections: {stats['schema rejected']}")
    (out_dir / "silver_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
