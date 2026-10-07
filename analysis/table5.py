"""Table 5: temporal faithfulness on the expert set.

A rewrite counts as faithful when P(faithful) is at least 0.5 (the verdict, for the LLM judges). The table gives balanced
accuracy and AUC overall, then by type: balanced accuracy with the recall on valid (V) and invalid (I) rewrites for
generalizations and increased precisions, and a single recall for reformulations (valid) and incorrect dates (invalid).
The two rows of Sect. 6.2, trained with MNLI pairs added, follow the English rows. Output: `results/tables/table5.{md,json}`."""
from __future__ import annotations

from tempolib import setup_console
from tempolib.evaluation import (
    TABLES_DIR, THRESHOLD, expert_sha, load_expert_pairs, load_results, write_json, write_text,
)
from tempolib.metrics import auc_ranks

ROWS = [
    ("English", "Baselines", "Zero-shot mDeBERTa", "probabilities/en.json", "mdeb_zs"),
    ("English", "Baselines", "Zero-shot MiniCheck", "probabilities/en.json", "rob_zs"),
    ("English", "Baselines", "Qwen3-4B", "llm_judge/qwen_qwen3-4b-2507__rules.json", None),
    ("English", "Baselines", "Qwen3-30B-A3B", "llm_judge/qwen_qwen3-30b-a3b-2507__rules.json", None),
    ("English", "FT: criterion + silver labels (ours)", "mDeBERTa", "probabilities/en.json", "mdeb_ft"),
    ("English", "FT: criterion + silver labels (ours)", "MiniCheck", "probabilities/en.json", "rob_ft"),
    ("English", "Ablation: criterion labels only", "mDeBERTa", "probabilities/en.json", "mdeb_rule_labels"),
    ("English", "Ablation: criterion labels only", "MiniCheck", "probabilities/en.json", "rob_rule_labels"),
    ("English", "Ablation: generator labels", "mDeBERTa", "probabilities/en.json", "mdeb_generator_labels"),
    ("English", "Ablation: generator labels", "MiniCheck", "probabilities/en.json", "rob_generator_labels"),
    ("English (Sect. 6.2)", "FT with 3,000 MNLI pairs added", "mDeBERTa", "probabilities/en.json", "mdeb_ft_mnli"),
    ("English (Sect. 6.2)", "FT with 3,000 MNLI pairs added", "MiniCheck", "probabilities/en.json", "rob_ft_mnli"),
    ("French", "Baseline", "mDeBERTa", "probabilities/fr.json", "mdeb_zs"),
    ("French", "FT (EN)", "mDeBERTa", "probabilities/fr.json", "mdeb_ft"),
    ("French", "FT (FR)", "mDeBERTa", "probabilities/fr.json", "mdeb_fr"),
    ("French", "FT (EN → FR)", "mDeBERTa", "probabilities/fr.json", "mdeb_fr_from_en"),
]
COLUMNS = ["BAcc", "AUC", "Gen BAcc", "Gen V", "Gen I", "Prec BAcc", "Prec V", "Prec I", "Reform V", "Incorr I"]
SET = {"English": ("expert", "en"), "English (Sect. 6.2)": ("expert", "en"), "French": ("expert_fr", "fr")}


def recall(pred, rows, cls, label):
    idx = [i for i, r in enumerate(rows) if r["error_type"] == cls and r["label"] == label]
    return sum(pred[i] == label for i in idx) / len(idx) if idx else None


def metrics(p, rows, with_auc):
    """The columns of Table 5 for the probabilities `p` of one model on the pairs `rows`."""
    y = [r["label"] for r in rows]
    pred = [int(q >= THRESHOLD) for q in p]
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    m = {"BAcc": (sum(pred[i] == 1 for i in pos) / len(pos) + sum(pred[i] == 0 for i in neg) / len(neg)) / 2,
         "AUC": auc_ranks(p, y) if with_auc else None}
    for name, cls in (("Gen", "generalization"), ("Prec", "precision")):
        v, i = recall(pred, rows, cls, 1), recall(pred, rows, cls, 0)
        m.update({f"{name} BAcc": (v + i) / 2, f"{name} V": v, f"{name} I": i})
    m["Reform V"] = recall(pred, rows, "reformulation", 1)
    m["Incorr I"] = recall(pred, rows, "wrong_date", 0)
    return m


def main() -> int:
    setup_console()
    sha = expert_sha()
    data = {}
    for section, group, label, source, key in ROWS:
        name, lang = SET[section]
        rows, _ = load_expert_pairs(lang)
        store = load_results(source)
        assert store["expert_sha"] == sha, f"{source} does not match the current expert set"
        p = store["probs"]["expert"] if key is None else store["probs"][name][key]
        data[(section, group, label)] = metrics(p, rows, with_auc=key is not None)

    def fmt(v):
        return "–" if v is None else f"{100 * v:.1f}"

    md = ["# Table 5 — Temporal faithfulness on the expert set (%)\n",
          f"Expert set sha {sha}. BAcc and AUC are computed on all decided pairs; the per-type columns follow Sect. 4.4: "
          "V = recall on valid rewrites, I = recall on invalid rewrites (rejected). Threshold P(faithful) ≥ 0.5.\n",
          "| | Model | " + " | ".join(COLUMNS) + " |", "|---|---|" + ":-:|" * len(COLUMNS)]
    for (section, group, label), m in data.items():
        md.append(f"| {section} — {group} | {label} | " + " | ".join(fmt(m[c]) for c in COLUMNS) + " |")
    write_text(TABLES_DIR / "table5.md", "\n".join(md) + "\n")
    write_json(TABLES_DIR / "table5.json",
               {"expert_sha": sha, "rows": [{"section": s, "group": g, "model": l, **m} for (s, g, l), m in data.items()]},
               indent=1)
    print("\n".join(md))
    print("\n-> results/tables/table5.{md,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
