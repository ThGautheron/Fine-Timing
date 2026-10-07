"""Sect. 6.2: what fine-tuning costs on general NLI (MNLI-m and SNLI, as entailment versus the rest).

It scores the zero-shot mDeBERTa, the fine-tuned one and the one trained with MNLI pairs added, and reports balanced
accuracy and the share of each gold class that is accepted. It needs the trained checkpoints and, in practice, a GPU.
Output: `results/retention/nli.json`."""
from __future__ import annotations

from tempolib import setup_console
from tempolib.evaluation import RETENTION_DIR, THRESHOLD, write_json
from tempolib.hf import Scorer
from tempolib.models import MODELS, checkpoint

KEYS = ["mdeb_zs", "mdeb_ft", "mdeb_ft_mnli"]
GOLD = ("entail", "neutral", "contra")


def nli_sets():
    from datasets import load_dataset

    sets = {"MNLI-m": load_dataset("nyu-mll/multi_nli", split="validation_matched"),
            "SNLI": load_dataset("stanfordnlp/snli", split="test")}
    return {k: [(r["premise"], r["hypothesis"], r["label"]) for r in d if r["label"] in (0, 1, 2)]
            for k, d in sets.items()}


def main() -> int:
    setup_console()
    keys = [k for k in KEYS if checkpoint(k) is not None]
    if not keys:
        raise SystemExit("no checkpoint found: train the models first (python run_pipeline.py --with train)")
    data = nli_sets()
    res = {}
    for key in keys:
        scorer = Scorer(checkpoint(key), 512, batch_size=64)
        for s, rows in data.items():
            probs = scorer.class_probabilities([(r[0], r[1]) for r in rows])
            p_ent = [p[scorer.entailment] for p in probs]
            gold3 = [r[2] for r in rows]
            y = [int(g == 0) for g in gold3]
            ok = [int(int(q >= THRESHOLD) == t) for q, t in zip(p_ent, y)]
            accept = {n: sum(int(q >= THRESHOLD) for q, g in zip(p_ent, gold3) if g == c) / max(1, gold3.count(c))
                      for c, n in enumerate(GOLD)}
            m = {"n": len(rows),
                 "bacc": (sum(o for o, t in zip(ok, y) if t) / sum(y)
                          + sum(o for o, t in zip(ok, y) if not t) / (len(y) - sum(y))) / 2,
                 "accept_rate": accept}
            res.setdefault(s, {})[key] = m
        scorer.release()
        print("  done:", key, flush=True)

    for s in data:
        print(f"\n=== {s} (n={res[s][keys[0]]['n']}) -- binary entailment vs rest, threshold {THRESHOLD} ===")
        print(f"{'':30}" + "".join(f"{k:>16}" for k in keys))
        print(f"{'BAcc':30}" + "".join(f"{res[s][k]['bacc']:16.3f}" for k in keys))
        for n in GOLD:
            print(f"{'% accepted | gold ' + n:30}" + "".join(f"{res[s][k]['accept_rate'][n]:16.3f}" for k in keys))

    write_json(RETENTION_DIR / "nli.json",
               {"threshold": THRESHOLD, "models": {k: MODELS[k].hub or MODELS[k].run for k in keys}, "metrics": res},
               indent=1)
    print("\n-> results/retention/nli.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
