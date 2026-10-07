"""Apply the criterion to every generated rewrite and report how often it agrees with the validity the generator was asked for."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tempolib import read_jsonl, resolve, setup_console, write_jsonl
from tempolib.judge import judge


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates", type=Path, help="candidate rewrites (JSON lines, optionally gzipped)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="judged candidates (JSON lines)")
    ap.add_argument("--lang", choices=["en", "fr"], default="en")
    args = ap.parse_args()

    rows = list(read_jsonl(resolve(args.candidates)))
    verdicts, agreement, classes, unjudged = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        verdict, rule_class, reason = judge(r, args.lang)
        r["rule_verdict"] = verdict
        r["rule_class"] = rule_class
        r["rule_reason"] = reason
        r["rule_label"] = None if verdict is None else (verdict == "valid")
        r["agrees_with_polarity"] = None if verdict is None else ((verdict == "valid") == (r["polarity"] == "valid"))
        r["agrees_with_llm"] = None if verdict is None else ((verdict == "valid") == bool(r["label_llm"]))
        verdicts[verdict] += 1
        if verdict is None:
            unjudged[reason.split(":")[0]] += 1
        else:
            agreement[(r["polarity"], verdict)] += 1
            classes[(r["error_type"], rule_class)] += 1
    write_jsonl(resolve(args.output), rows)

    n = sum(v for k, v in verdicts.items() if k)
    print(f"candidates: {len(rows)} | judged: {n} | not judged: {verdicts[None]}")
    print("verdicts:", {k: v for k, v in verdicts.items() if k})
    print("\nreasons for not judging:", dict(unjudged))
    print("\nintended polarity x rule verdict:")
    for pol in ["valid", "invalid"]:
        tot = sum(agreement[(pol, v)] for v in ["valid", "tolerated", "invalid"])
        if not tot:
            continue
        print(f"  {pol:8s} n={tot:6d} -> " + " | ".join(
            f"{v} {agreement[(pol, v)]:5d} ({100 * agreement[(pol, v)] / tot:4.1f}%)"
            for v in ["valid", "tolerated", "invalid"]))
    judged = [r for r in rows if r["agrees_with_polarity"] is not None]
    print(f"\nrule / intended polarity agreement: "
          f"{sum(r['agrees_with_polarity'] for r in judged)}/{len(judged)} = "
          f"{100 * sum(r['agrees_with_polarity'] for r in judged) / len(judged):.1f}%")
    print(f"rule / generator label agreement  : "
          f"{sum(r['agrees_with_llm'] for r in judged)}/{len(judged)} = "
          f"{100 * sum(r['agrees_with_llm'] for r in judged) / len(judged):.1f}%")
    print("\nrequested type -> class deduced from the interval relation:")
    for k in sorted(classes, key=str):
        if classes[k] > 30:
            print(f"  {k[0]:16s} -> {k[1]:16s} {classes[k]}")
    print(f"\n-> {resolve(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
