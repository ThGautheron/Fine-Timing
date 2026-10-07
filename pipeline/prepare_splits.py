"""Split the annotated anchors by source article into train, dev and test.

The split keeps the anchor types balanced across the three parts. Only train and dev are rewritten and used for training."""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from tempolib import load_config, resolve, setup_console, write_jsonl
from tempolib.anchors import load_all
from tempolib.splits import ordered_types, search_partition


def _partition(by_doc: dict, types: list[str], ratio: float, seed: int, n_trials: int) -> set:
    if len(by_doc) < 3 or ratio <= 0:
        return set()
    rng = random.Random(seed)
    total = sum(len(v) for v in by_doc.values())
    return set(search_partition(by_doc, types, ratio, total, rng, n_trials))


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=Path("configs/data_en.yaml"))
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    sp = cfg["split"]
    splits_dir = resolve(cfg["paths"]["splits_dir"])

    anchors = load_all(cfg["anchors"])
    if not anchors:
        print("No anchor loaded: check `anchors` in the configuration.")
        return 1

    by_doc: dict[str, list] = defaultdict(list)
    for a in anchors:
        by_doc[a.doc_id].append(a)
    types = ordered_types({a.error_type for a in anchors})
    by_doc_types = {d: [{"error_type": a.error_type} for a in v] for d, v in by_doc.items()}
    print(f"{len(anchors)} anchors - {len(by_doc)} documents - types = {types}")

    test_docs = _partition(by_doc_types, types, sp["test_ratio"], sp["seed"], sp["n_trials"])
    remaining = {d: v for d, v in by_doc_types.items() if d not in test_docs}
    dev_ratio_adj = sp["dev_ratio"] / max(1e-6, 1 - sp["test_ratio"])
    dev_docs = _partition(remaining, types, dev_ratio_adj, sp["seed"] + 1, sp["n_trials"])

    assign = {d: "test" for d in test_docs}
    assign.update({d: "dev" for d in dev_docs})

    groups: dict[str, list] = {"train": [], "dev": [], "test": []}
    for a in anchors:
        row = a.as_dict()
        row["split"] = assign.get(a.doc_id, "train")
        groups[row["split"]].append(row)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in groups.items():
        write_jsonl(splits_dir / f"anchors_{split}.jsonl", rows)
        by_type = dict(sorted(Counter(r["error_type"] for r in rows).items()))
        print(f"[{split}] {len(rows):4} anchors, {len({r['doc_id'] for r in rows})} documents  {by_type}")
    print(f"-> {splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
