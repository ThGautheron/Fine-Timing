"""Table 3: one source sentence, seven rewrites of its date, and what each judge thinks of them.

The criterion, the zero-shot mDeBERTa and MiniCheck (faithful if the entailment probability is at least 0.5) and the
Qwen3-30B-A3B judge all read the source sentence as the document and the rewritten sentence as the claim. The Qwen
verdicts are cached, so no server is needed to rerun it. Output: `results/tables/table3.{md,json}`."""
from __future__ import annotations

import argparse

from analysis.llm_judge import judge_pairs
from tempolib import setup_console
from tempolib.evaluation import JUDGE_DIR, TABLES_DIR, THRESHOLD, write_json, write_text
from tempolib.hf import Scorer
from tempolib.judge import EN, judge
from tempolib.models import MODELS
from tempolib.tiers import parse

SOURCE = "The treaty was signed in 1294."
REWRITES = ["the 1290s", "the 13th century", "the High Middle Ages", "the Middle Ages", "the summer of 1294",
            "anno Domini 1294", "1295"]
LLM = "qwen/qwen3-30b-a3b-2507"


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    args = ap.parse_args()

    claims = [SOURCE.replace("in 1294", f"in {expr}") for expr in REWRITES]
    criterion = []
    for expr, claim in zip(REWRITES, claims):
        verdict, kind, _reason = judge(
            {"anchor_expression": "1294", "original_sentence": SOURCE, "claim": claim, "document": SOURCE,
             "interval_anchor": (1294, 1294), "granularity": "year"}, EN)
        criterion.append((verdict, kind, parse(expr)["tier"]))

    pairs = [(SOURCE, c) for c in claims]
    nli = {}
    for key in ("mdeb_zs", "rob_zs"):
        scorer = Scorer(MODELS[key].hub, MODELS[key].max_length)
        nli[key] = scorer(pairs)
        scorer.release()
    llm = judge_pairs(LLM, pairs, args.base_url, JUDGE_DIR / "table3_qwen_qwen3-30b-a3b-2507__rules.jsonl")

    def mark(ok):
        return "✓" if ok else "✗"

    rows = []
    for i, expr in enumerate(REWRITES):
        verdict, kind, tier = criterion[i]
        rows.append({"rewrite": expr, "type": kind, "granularity": tier, "criterion": verdict,
                     "mdeberta": nli["mdeb_zs"][i], "minicheck": nli["rob_zs"][i], "qwen": llm[i]["verdict"]})
    md = [f"# Table 3 — Verdicts on rewrites of «{SOURCE}»\n",
          "| Rewrite of 1294 | Type | Gran. | Criterion | mDeBERTa | MiniCheck | Qwen3-30B-A3B |",
          "|---|---|---|:-:|:-:|:-:|:-:|"]
    for r in rows:
        md.append(f"| {r['rewrite']} | {r['type']} | {r['granularity']} | {mark(r['criterion'] == 'valid')} | "
                  f"{mark(r['mdeberta'] >= THRESHOLD)} ({r['mdeberta']:.2f}) | "
                  f"{mark(r['minicheck'] >= THRESHOLD)} ({r['minicheck']:.2f}) | {mark(r['qwen'] == 'faithful')} |")
    write_text(TABLES_DIR / "table3.md", "\n".join(md) + "\n")
    write_json(TABLES_DIR / "table3.json", rows, indent=1)
    print("\n".join(md))
    print("\n-> results/tables/table3.{md,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
