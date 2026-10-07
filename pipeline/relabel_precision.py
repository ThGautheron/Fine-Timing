"""Apply the criterion to the precisions taken from real summaries.

A precision is faithful only if the source states a date inside the narrower interval, so the validity pre-filled
in the annotation is replaced by this check."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tempolib import read_jsonl, resolve, setup_console, write_jsonl
from tempolib.judge import get_language, is_attested


def _expression(r: dict) -> str:
    return r.get("anchor_expression_claim") or r.get("anchor_expression") or ""


def strict_label(r: dict, lang):
    p = lang.parse(_expression(r))
    return None if p is None else int(is_attested(p, r["document"], lang))


def relabel(in_dir: Path, out_dir: Path, lang) -> None:
    """Rewrite the labels of the human precision rows in both splits and save the result."""
    report = [f"relabel_precision: {in_dir.name} -> {out_dir.name}"]
    for split in ("train", "dev"):
        rows = list(read_jsonl(in_dir / f"{split}.jsonl"))
        c = Counter()
        for r in rows:
            if r.get("origin") != "human" or r.get("error_type") != "precision":
                continue
            new = strict_label(r, lang)
            if new is None:
                c["unparsable (label kept)"] += 1
                continue
            r["verification"]["label_prefill"] = r["label"]
            r["verification"]["relabel"] = "expert_criterion"
            c[f"{r['label']}->{new}"] += 1
            r["label"] = new
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        pos = sum(r["label"] for r in rows)
        report.append(f"[{split}] human precision: {dict(c)} | total {len(rows)} (pos={pos}/neg={len(rows) - pos})")
    (out_dir / "relabel_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_dir", type=Path, required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "fr"], default="en")
    a = ap.parse_args()
    relabel(resolve(a.in_dir), resolve(a.out_dir), get_language(a.lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
