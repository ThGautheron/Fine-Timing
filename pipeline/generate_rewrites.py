"""Ask an LLM to rewrite the date of every faithful train/dev anchor (Sect. 4.2).

For each anchor, one prompt per type and validity asks for a few rewrites of the date only. Replies are cached,
parsed, and filtered to drop rewrites that changed anything else. The labelling itself is left to the judge."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

from tempolib import extraction
from tempolib import load_config, now_iso, read_jsonl, resolve, setup_console, write_jsonl
from tempolib import prompts as P
from tempolib.cleaning import clean_chunk, ensure_contains
from tempolib.humanset import is_citation_year
from tempolib.intervals import anchor_interval, granularity_level
from tempolib.llm import OpenAICompatibleClient, parse_variants
from tempolib.textclean import clean_claim, window_document

CLASSES = ["generalization", "precision", "reformulation", "wrong_date"]
GENERALIZATION_SEED_CLASSES = ["generalization", "precision"]

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_LEAK = re.compile(
    r'"""|\bladder\b|\bfaithful\b|\brewrite[sd]?\b|\brung\b|\bpolarit|^\s*\[|^\s*\d+[.)]\s|→|\bJSON\b',
    re.I,
)


def _wellformed(claim: str) -> str | None:
    """Why a rewrite looks malformed (too short, no final punctuation, prompt leak...), or None if it is fine."""
    c = (claim or "").strip()
    if len(c) < 10:
        return "too short"
    if not re.search(r"[.!?…][\"»'\)\]]?\s*$", c):
        return "no final punctuation"
    if _LEAK.search(c):
        return "prompt leak / formatting"
    if re.search(r"  +", c) or c.count('"') % 2 == 1:
        return "multiple spaces / unbalanced quotes"
    return None


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def _strip_temporal(text: str, backend: ModuleType) -> str:
    for _iv, _g, surface in backend.all_intervals(text):
        text = text.replace(surface, " ")
    return " ".join(w.lower() for w in _WORD.findall(re.sub(r"\d+", " ", text)))


def _content_preserved(original: str, claim: str, backend: ModuleType, min_jaccard: float = 0.78) -> bool:
    """Whether the rewrite changed (almost) nothing but its date: same words once the dates are removed."""
    lo, lc = len(original), len(claim)
    if lc < 0.6 * lo or lc > 1.6 * lo:
        return False
    a = set(_strip_temporal(original, backend).split())
    b = set(_strip_temporal(claim, backend).split())
    if not a:
        return True
    return len(a & b) / len(a | b) >= min_jaccard


def build_jobs(anchors: list[dict], strip_bib: bool, doc_window: int, backend: ModuleType) -> list[tuple]:
    """One (anchor, type, validity) job per prompt to send: each faithful anchor gets a series per type and validity."""
    jobs = []
    for a in anchors:
        anchor = anchor_interval(a["expression"], a["sentence"], backend)
        if anchor is None:
            continue
        interval, granularity = anchor
        seed_classes = a.get("_seed_classes", CLASSES)
        source_window = a.get("source_window", "") or ""
        document = ensure_contains(
            clean_chunk(a["source_chunk"], strip_bibliography=strip_bib, noise=backend.NOISE), source_window
        )
        if not document.strip():
            continue
        if doc_window:
            focus = source_window if source_window not in ("", "(no match)") else a["sentence"]
            document = window_document(document, focus, max_tokens=doc_window, segment=backend.segment)
        sentence_clean = clean_claim(a["sentence"])
        if a["expression"] and a["expression"] not in sentence_clean:
            sentence_clean = a["sentence"]
        base = {
            "anchor_item_id": a["item_id"], "doc_id": a["doc_id"], "split": a["split"],
            "anchor_expression": a["expression"], "granularity": granularity,
            "interval_anchor": list(interval), "original_sentence": sentence_clean, "document": document,
            "source_file": a.get("source_file"), "source_pub_year": a.get("source_pub_year"),
            "stratum": a.get("stratum"), "seed_error_type": a.get("error_type", "none"),
        }
        skip_generalization = (granularity_level(interval, granularity) or 0) >= 6
        for kind in seed_classes:
            if kind == "generalization" and skip_generalization:
                continue
            for validity in P.polarities_for(kind):
                jobs.append((base, kind, validity))
    return jobs


def _candidates_from_raw(jobs, raw_dir: Path, llm_cfg: dict, client, n: int, regenerate: bool,
                         stats: Counter, lang: str, backend: ModuleType) -> list[dict]:
    """Run (or reload from the cache) every prompt, parse the replies and keep the well-formed rewrites."""
    candidates = []
    model_tag = _safe(llm_cfg["model"])
    seed = llm_cfg["seed"]
    for base, kind, validity in jobs:
        key = _safe(f"{model_tag}__{base['anchor_item_id']}__{kind}__{validity}")
        raw_fp = raw_dir / f"{key}.json"

        if raw_fp.exists() and not regenerate:
            payload = json.loads(raw_fp.read_text(encoding="utf-8"))
            stats["reused"] += 1
        elif client is None:
            stats["missing_raw"] += 1
            continue
        else:
            prompt = P.build(kind, validity, anchor_expr=base["anchor_expression"],
                             sentence=base["original_sentence"], source=base["document"],
                             n=n, source_pub_year=base["source_pub_year"], lang=lang)
            try:
                res = client.complete(prompt, seed=seed)
            except Exception as exc:
                stats["llm_error"] += 1
                print(f"  ! {key}: {exc}")
                continue
            payload = {"prompt": prompt, "response": res["content"], "model": res["model"],
                       "attempt": res["attempt"], "generated_at": now_iso()}
            raw_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            stats["called"] += 1

        variants = parse_variants(payload["response"])
        if not variants:
            stats["parse_fail"] += 1
            continue
        variants = [c for c in (clean_claim(v) for v in variants) if c]
        stats["variants"] += len(variants)
        for claim in variants:
            if not _content_preserved(base["original_sentence"], claim, backend):
                stats["reject: non-temporal content modified"] += 1
                continue
            bad = _wellformed(claim)
            if bad:
                stats[f"reject: {bad}"] += 1
                continue
            candidates.append({
                **base, "error_type": kind, "polarity": validity,
                "label_llm": P.label_llm_for(validity), "claim": claim,
                "llm_model": payload["model"], "llm_seed": seed,
                "prompt_version": P.PROMPT_VERSION, "generated_at": payload["generated_at"],
            })
    return candidates


def select_anchors(splits_dir: Path) -> tuple[list[dict], list[dict], int]:
    """The faithful train/dev anchors to rewrite: unchanged ones and valid generalizations, without citation years."""
    none_anchors, gen_anchors = [], []
    dropped = 0
    for split in ("train", "dev"):
        fp = splits_dir / f"anchors_{split}.jsonl"
        if not fp.exists():
            raise SystemExit(f"missing: {fp} - run pipeline.prepare_splits first")
        for r in read_jsonl(fp):
            kind = r.get("error_type")
            if kind == "none":
                bucket = none_anchors
            elif kind == "generalization" and (r.get("validity") or "").strip() == "valid":
                r = dict(r)
                r["_seed_classes"] = GENERALIZATION_SEED_CLASSES
                bucket = gen_anchors
            else:
                continue
            if is_citation_year(r):
                dropped += 1
                continue
            bucket.append(r)
    return none_anchors, gen_anchors, dropped


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=Path("configs/data_en.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="maximum number of `none` anchors (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    ap.add_argument("--regenerate", action="store_true", help="ignore the raw reply cache")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not call the LLM: rebuild the candidates from the cached raw replies")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    lang = cfg["language"]
    backend = extraction.get_backend(lang)
    llm_cfg = cfg["llm"]
    n = int(llm_cfg["n_per_series"])
    strip_bib = bool(cfg["preprocessing"].get("strip_bibliography", True))
    doc_window = int(cfg["preprocessing"].get("doc_window_tokens", 0))
    splits_dir = resolve(cfg["paths"]["splits_dir"])
    gen_dir = resolve(cfg["paths"]["generation_dir"])
    raw_dir = gen_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    none_anchors, gen_anchors, dropped = select_anchors(splits_dir)
    if args.limit:
        none_anchors = none_anchors[: args.limit]
    anchors = none_anchors + gen_anchors

    jobs = build_jobs(anchors, strip_bib, doc_window, backend)
    print(f"{len(anchors)} faithful anchors kept (train+dev): {len(none_anchors)} 'none' + "
          f"{len(gen_anchors)} generalization seeds - {len(jobs)} prompt series "
          f"- {dropped} anchors dropped (citation year).")
    if not jobs:
        return 1

    if args.dry_run:
        base, kind, validity = jobs[0]
        print(f"\n--- example: {kind} / {validity} ---\n")
        print(P.build(kind, validity, anchor_expr=base["anchor_expression"],
                      sentence=base["original_sentence"], source=base["document"],
                      n=n, source_pub_year=base["source_pub_year"], lang=lang))
        return 0

    client = None
    if not args.verify_only:
        client = OpenAICompatibleClient(
            base_url=llm_cfg["base_url"], model=llm_cfg["model"], api_key=llm_cfg.get("api_key", "EMPTY"),
            temperature=llm_cfg["temperature"], seed=llm_cfg["seed"],
            timeout_s=llm_cfg["timeout_s"], max_retries=llm_cfg["max_retries"],
        )

    stats = Counter()
    candidates = _candidates_from_raw(jobs, raw_dir, llm_cfg, client, n, args.regenerate, stats, lang, backend)
    write_jsonl(gen_dir / "candidates.jsonl", candidates)
    print(f"\ngeneration: {dict(stats)}")
    (gen_dir / "generation_stats.json").write_text(
        json.dumps({"generation": dict(stats)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(candidates)} candidates -> {gen_dir / 'candidates.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
