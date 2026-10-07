"""Give every model's P(faithful) for each pair of the expert set.

English models read the English wording, the French ones the French wording. A model with a local checkpoint
(`outputs/models/<run>` or `$MODELS_DIR/<run>`) is run; otherwise the probabilities of the reported checkpoint, stored in
`data/reference/`, are used. Output: `results/probabilities/{en,fr}.json`."""
from __future__ import annotations

import argparse

from tempolib import setup_console
from tempolib.evaluation import (
    PROBABILITIES_DIR, THRESHOLD, expert_sha, load_expert_pairs, probability_store, write_json,
)
from tempolib.hf import Scorer
from tempolib.models import MODELS, checkpoint

EXPERIMENTS = {
    "en": ("expert", ["mdeb_zs", "rob_zs", "mdeb_ft", "rob_ft", "mdeb_rule_labels", "mdeb_generator_labels",
                      "rob_rule_labels", "rob_generator_labels", "mdeb_ft_mnli", "rob_ft_mnli"]),
    "fr": ("expert_fr", ["mdeb_zs", "mdeb_ft", "mdeb_fr", "mdeb_fr_from_en"]),
}


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference-only", action="store_true",
                    help="never load a checkpoint: use the cached reference probabilities")
    args = ap.parse_args()

    sha = expert_sha()
    for lang, (name, keys) in EXPERIMENTS.items():
        pairs, _ = load_expert_pairs(lang)
        rows = [(r["document"], r["claim"]) for r in pairs]
        store = probability_store(f"probabilities_{lang}.json")
        if store.get("expert_sha") not in (None, sha):
            store["probs"] = {}
        store["expert_sha"], store["threshold"] = sha, THRESHOLD
        store["probs"].setdefault(name, {})
        sources = {}
        for key in keys:
            spec = MODELS[key]
            path = None if args.reference_only else checkpoint(spec)
            cached = key in store["probs"][name]
            if spec.hub and cached:
                path = None
            if path is None and not cached:
                print(f"  [{lang}] missing: {key} (no checkpoint and no cached probabilities)")
                continue
            if path is None:
                sources[key] = "reference"
                print(f"  [{lang}] cached probabilities: {key}")
                continue
            scorer = Scorer(path, spec.max_length)
            store["probs"][name][key] = scorer(rows)
            scorer.release()
            sources[key] = "checkpoint"
            print(f"  [{lang}] scored: {key} ({path})")
        store["sources"] = sources
        write_json(PROBABILITIES_DIR / f"{lang}.json", store)
        print(f"-> results/probabilities/{lang}.json ({len(rows)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
