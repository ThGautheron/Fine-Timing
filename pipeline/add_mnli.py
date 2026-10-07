"""Add MNLI pairs to a training set, so that the model also sees content errors (Sect. 6.2).

Train gets `--n-train` pairs, half entailment (faithful) and the rest neutral or contradiction (unfaithful); dev gets
`--n-dev` pairs from the mismatched validation split."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tempolib import read_jsonl, resolve, setup_console, write_jsonl

MNLI_LABELS = ["entailment", "neutral", "contradiction"]


def mnli_rows(split: str, n: int, rng: random.Random, target_split: str) -> list[dict]:
    from datasets import load_dataset

    d = load_dataset("nyu-mll/multi_nli", split=split)
    by = {0: [], 1: [], 2: []}
    for r in d:
        if r["label"] in by and len(r["premise"]) > 20 and len(r["hypothesis"]) > 5:
            by[r["label"]].append(r)
    take = {0: n // 2, 1: n // 4, 2: n - n // 2 - n // 4}
    rows = []
    for lab, k in take.items():
        for r in rng.sample(by[lab], k):
            rows.append({"document": r["premise"], "claim": r["hypothesis"], "label": int(lab == 0),
                         "error_type": "content", "origin": "mnli", "id": f"mnli-{split}-{r['pairID']}",
                         "doc_id": f"mnli:{r['promptID']}", "split": target_split,
                         "granularity": None, "anchor_expression": None, "original_sentence": None,
                         "verification": {"mnli_label": MNLI_LABELS[lab]}})
    return rows


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="training set to extend")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=3000, help="MNLI pairs added to train (half entailment)")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src, out = resolve(a.src), resolve(a.out)

    for split, mnli_split, n in (("train", "train", a.n_train), ("dev", "validation_mismatched", a.n_dev)):
        base = list(read_jsonl(src / f"{split}.jsonl"))
        add = mnli_rows(mnli_split, n, rng, "train" if split == "train" else "dev")
        rows = base + add
        rng.shuffle(rows)
        write_jsonl(out / f"{split}.jsonl", rows)
        pos = sum(r["label"] for r in rows)
        print(f"{split:5} {len(base):6d} base + {len(add):5d} MNLI = {len(rows):6d} rows | pos {pos} neg {len(rows) - pos} "
              f"| MNLI: {100 * len(add) / len(rows):.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
