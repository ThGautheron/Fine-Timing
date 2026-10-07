"""Put the annotated anchors and the labelled rewrites together, remove duplicates, and balance valid and invalid pairs."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from tempolib import __version__, load_config, read_jsonl, resolve, setup_console, write_jsonl
from tempolib.dedup import drop_exact
from tempolib.humanset import build_rows as build_human_rows
from tempolib.schema import new_row, validate
from tempolib.textclean import clean_claim, window_document


def _to_row(c: dict, doc_window: int = 0) -> dict | None:
    """A labelled rewrite as a dataset row, or None when it has no label."""
    label = c["verification"].get("label")
    if label is None:
        return None
    claim = clean_claim(c["claim"])
    if c.get("anchor_expression") and c["anchor_expression"] not in claim:
        claim = c["claim"]
    document = c["document"]
    if doc_window:
        document = window_document(document, c.get("original_sentence", ""), max_tokens=doc_window)
    verification = {k: v for k, v in c["verification"].items() if k != "label"}
    return new_row(
        document=document, claim=claim, label=int(bool(label)), error_type=c["error_type"],
        granularity=c.get("granularity"), anchor_expression=c["anchor_expression"],
        original_sentence=c["original_sentence"], doc_id=c["doc_id"],
        source_pub_year=c.get("source_pub_year"), stratum=c.get("stratum"),
        split=c.get("split", "train"), origin="synthetic", verification=verification,
        provenance={"llm_model": c.get("llm_model"), "llm_seed": c.get("llm_seed"),
                    "prompt_version": c.get("prompt_version"),
                    "tempolib_version": __version__, "generated_at": c.get("generated_at")},
    )


def balance(rows: list[dict], ratio: float | None, rng: random.Random) -> list[dict]:
    """Keep all valid pairs and subsample the invalid ones to `ratio` per valid pair."""
    if ratio is None:
        return rows
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    target = round(ratio * len(pos)) if pos else len(neg)
    if len(neg) <= target:
        return rows
    rng.shuffle(neg)
    return pos + neg[:target]


def _dist(rows: list[dict], field: str) -> str:
    c = Counter(r[field] for r in rows)
    return ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda x: str(x[0])))


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/data_en.yaml"))
    ap.add_argument("--labelled", type=Path, required=True, help="labelled rewrites (pipeline.build_labelled)")
    ap.add_argument("--out", type=Path, required=True, help="output dataset directory")
    ap.add_argument("--no-human", action="store_true", help="generated rewrites only (no human anchors)")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    ac = cfg["assemble"]
    strip_bib = bool(cfg["preprocessing"].get("strip_bibliography", True))
    doc_window = int(cfg["preprocessing"].get("doc_window_tokens", 0))
    use_human = ac.get("include_human_anchors", True) and not args.no_human
    keep_human_positives = bool(ac.get("include_human_positives", True)) and not args.no_human
    splits_dir = resolve(cfg["paths"]["splits_dir"])
    out_dir = resolve(args.out)
    rng = random.Random(ac["seed"])
    labelled_fp = resolve(args.labelled)
    if not labelled_fp.exists():
        print(f"missing: {labelled_fp} - run pipeline.build_labelled first")
        return 1
    labelled = list(read_jsonl(labelled_fp))

    human_rows: list[dict] = []
    human_stats: dict[str, dict] = {}
    if use_human:
        for split in ("train", "dev"):
            fp = splits_dir / f"anchors_{split}.jsonl"
            if not fp.exists():
                continue
            hr, hs = build_human_rows(
                read_jsonl(fp), split=split, strip_bib=strip_bib, tempolib_version=__version__,
                keep_positive=keep_human_positives, doc_window=doc_window,
            )
            human_rows += hr
            human_stats[split] = hs

    rows = human_rows + [r for r in (_to_row(c, doc_window) for c in labelled) if r is not None]
    rows, n_exact = drop_exact(rows)

    final: dict[str, list[dict]] = {"train": [], "dev": []}
    for r in rows:
        r["split"] = r["split"] if r["split"] in ("train", "dev") else "train"
        final[r["split"]].append(r)

    rep = ["=" * 70, "DATASET ASSEMBLY", "=" * 70, ""]
    if use_human:
        for split, hs in human_stats.items():
            rep.append(f"human anchors [{split}]: {hs.get('rows', 0)} rows "
                       f"(pos={hs.get('pos', 0)} neg={hs.get('neg', 0)}, merged={hs.get('merged', 0)})")
    else:
        rep.append("human anchors: disabled")
    rep.append(f"exact deduplication: -{n_exact}")
    rep.append("")

    invalid = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        balanced = balance(final[split], ac.get("neg_pos_ratio"), rng)
        clean = [r for r in balanced if not validate(r)]
        invalid += len(balanced) - len(clean)
        rng.shuffle(clean)
        write_jsonl(out_dir / f"{split}.jsonl", clean)
        pos = sum(1 for r in clean if r["label"] == 1)
        rep += [f"[{split}]  {len(clean)} rows   pos={pos}  neg={len(clean) - pos}",
                f"    error_type  : {_dist(clean, 'error_type')}",
                f"    granularity : {_dist(clean, 'granularity')}",
                f"    stratum     : {_dist(clean, 'stratum')}",
                f"    origin      : {_dist(clean, 'origin')}",
                f"    documents   : {len({r['doc_id'] for r in clean})}", ""]
    if invalid:
        rep.append(f"!! {invalid} rows dropped (invalid schema).")

    text = "\n".join(rep)
    (out_dir / "assemble_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
