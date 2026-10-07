"""Sect. 6.2: what fine-tuning costs on grounded fact checking (Reveal and FactCheck-GPT from LLM-AggreFact).

It uses a third-party copy of the test set on the Hugging Face Hub, since the official one is gated. It needs the
trained checkpoints and, in practice, a GPU. Output: `results/retention/aggrefact.json`."""
from __future__ import annotations

from tempolib import setup_console
from tempolib.evaluation import RETENTION_DIR, write_json
from tempolib.hf import Scorer
from tempolib.models import checkpoint

KEYS = ["mdeb_zs", "mdeb_ft", "rob_zs", "rob_ft", "mdeb_ft_mnli", "rob_ft_mnli"]
SUBSETS = ["Reveal", "FactCheck-GPT"]
DATASET = "NinaCalvi/llm_aggrefact_pre_aug9"


def main() -> int:
    from datasets import load_dataset

    setup_console()
    if not any(checkpoint(k) for k in KEYS if not k.endswith("_zs")):
        raise SystemExit("no fine-tuned checkpoint found: train the models first (python run_pipeline.py --with train)")
    d = load_dataset(DATASET, split="test")
    rows = [{"evidence": r["doc"], "claim": r["claim"], "label": "SUPPORTS" if str(r["label"]) == "1" else "NOT SUPPORTED",
             "subset": r["dataset"]} for r in d if r["dataset"] in SUBSETS]
    print(f"LLM-AggreFact (third-party copy) test: {len(rows)} pairs; "
          + ", ".join(f"{k} {sum(r['subset'] == k for r in rows)}" for k in SUBSETS), flush=True)
    y = [int(r["label"] == "SUPPORTS") for r in rows]
    pairs = [(r["evidence"], r["claim"]) for r in rows]

    P = {}
    for key in KEYS:
        path = checkpoint(key)
        if path is None:
            continue
        scorer = Scorer(path, 512, batch_size=32, sort_key=lambda pair: len(pair[0]))
        P[key] = scorer(pairs)
        scorer.release()
        print("  done:", key, flush=True)

    res = {}
    for k, p in P.items():
        res[k] = {}
        for sub in SUBSETS + ["macro"]:
            idx = [i for i, r in enumerate(rows) if sub == "macro" or r["subset"] == sub]
            if sub == "macro":
                m = {"bacc": sum(res[k][x]["bacc"] for x in SUBSETS) / len(SUBSETS)}
            else:
                pos = [i for i in idx if y[i]]
                neg = [i for i in idx if not y[i]]
                accepted = sum(p[i] >= 0.5 for i in pos) / len(pos)
                rejected = sum(p[i] < 0.5 for i in neg) / len(neg)
                m = {"bacc": (accepted + rejected) / 2, "supported_accepted": accepted,
                     "unsupported_rejected": rejected, "n": len(idx), "pos": len(pos)}
            res[k][sub] = m
    print()
    print(f"{'':14}" + "".join(f"{sub:>14}" for sub in SUBSETS + ["macro"]))
    for k, ms in res.items():
        print(f"{k:14}" + "".join(f"{100 * ms[sub]['bacc']:14.1f}" for sub in SUBSETS + ["macro"]))
    write_json(RETENTION_DIR / "aggrefact.json",
               {"dataset": f"{DATASET} (third-party copy of the LLM-AggreFact test set)", "subsets": SUBSETS,
                "metrics": res}, indent=1)
    print("-> results/retention/aggrefact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
