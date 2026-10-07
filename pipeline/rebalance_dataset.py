"""Fix two imbalances of the generated rewrites.

First, "between X and Y" precisions are too frequent among the invalid pairs, so some are dropped. Second, the Victorian
era is over-represented as a named period, so it is replaced by other periods whenever the judge gives the new claim
the same label."""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from tempolib import read_jsonl, resolve, setup_console, write_jsonl
from tempolib.judge import judge

PATTERN_BETWEEN = re.compile(r"\bbetween\s+\d{3,4}\s+and\s+\d{3,4}\b", re.I)
VICTORIAN_RE = re.compile(r"(the\s+)?victorian\s+(?:era|period|age)", re.I)

VICTORIAN_CANDIDATES = [
    "Georgian era", "Regency period", "Edwardian era", "Belle Epoque",
    "Gilded Age", "Progressive Era", "interwar period", "Romantic era",
]


def is_between_precision(row: dict, label: int) -> bool:
    return (row.get("error_type") == "precision" and row["label"] == label
            and bool(PATTERN_BETWEEN.search(row["claim"])))


def rebalance_between_precision(rows: list[dict], rng: random.Random,
                                target_valid_share: float = 0.56) -> tuple[list[dict], dict]:
    """Drop invalid "between X and Y" precisions until about 56 % of that group is valid."""
    invalid_ids = [i for i, r in enumerate(rows) if is_between_precision(r, 0)]
    n_valid = sum(is_between_precision(r, 1) for r in rows)
    n_invalid = len(invalid_ids)
    if n_invalid == 0 or n_valid == 0:
        return rows, {"n_invalid_before": n_invalid, "n_dropped": 0, "n_invalid_after": n_invalid}

    invalid_to_keep = round(n_valid * (1 - target_valid_share) / target_valid_share)
    n_drop = max(0, n_invalid - invalid_to_keep)
    dropped = set(rng.sample(invalid_ids, n_drop)) if n_drop else set()

    kept = [r for i, r in enumerate(rows) if i not in dropped]
    return kept, {
        "n_valid_same_bucket": n_valid,
        "n_invalid_before": n_invalid,
        "n_dropped": len(dropped),
        "n_invalid_after": n_invalid - len(dropped),
    }


def substitute_victorian(claim: str, candidate: str) -> str | None:
    """The claim with "the Victorian era" replaced by `candidate`, or None if it does not mention it."""
    m = VICTORIAN_RE.search(claim)
    if not m:
        return None
    article = m.group(1) or ""
    replacement = article + candidate
    if m.start() == 0:
        replacement = replacement[0].upper() + replacement[1:]
    return claim[: m.start()] + replacement + claim[m.end():]


def _verdict_for(row: dict, claim: str) -> str | None:
    """The criterion's verdict on `claim`, as a rewrite of the same anchor and document as `row`."""
    r = {
        "original_sentence": row["original_sentence"],
        "claim": claim,
        "interval_anchor": row["verification"]["interval_anchor"],
        "granularity": row.get("granularity"),
        "document": row["document"],
    }
    verdict, _cls, _reason = judge(r)
    return verdict


def diversify_victorian(rows: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Replace "the Victorian era" by another period, keeping a replacement only if the label stays the same."""
    victorian_ids = [i for i, r in enumerate(rows) if "victorian" in r["claim"].lower()]
    rng.shuffle(victorian_ids)
    periods = list(VICTORIAN_CANDIDATES)
    rng.shuffle(periods)

    n_replaced = 0
    per_period = Counter()

    for k, i in enumerate(victorian_ids):
        row = rows[i]
        if not row.get("verification", {}).get("interval_anchor"):
            continue
        start = k % len(periods)
        for period in periods[start:] + periods[:start]:
            new_claim = substitute_victorian(row["claim"], period)
            if new_claim is None:
                break
            verdict = _verdict_for(row, new_claim)
            if verdict is not None and int(verdict == "valid") == row["label"]:
                row["claim"] = new_claim
                per_period[period] += 1
                n_replaced += 1
                break

    return rows, {
        "n_victorian_total": len(victorian_ids),
        "n_replaced": n_replaced,
        "n_kept_original": len(victorian_ids) - n_replaced,
        "per_candidate": dict(per_period),
    }


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    args = ap.parse_args()

    in_dir, out_dir = resolve(args.in_dir), resolve(args.out_dir)
    report = [f"rebalance_dataset: {args.in_dir} -> {args.out_dir} (seed={args.seed})\n"]

    for split in ("train", "dev"):
        rng = random.Random(args.seed)
        rows = list(read_jsonl(in_dir / f"{split}.jsonl"))
        n0 = len(rows)

        rows, stat_a = rebalance_between_precision(rows, rng)
        rows, stat_b = diversify_victorian(rows, rng)

        write_jsonl(out_dir / f"{split}.jsonl", rows)
        lines = [f"[{split}] {n0} -> {len(rows)} rows",
                 f"  fix A (precision between-X-and-Y): {stat_a}",
                 f"  fix B (victorian -> diversified) : {stat_b}"]
        report += lines
        print("\n".join(lines))

    (out_dir / "rebalance_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n-> {out_dir / 'rebalance_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
