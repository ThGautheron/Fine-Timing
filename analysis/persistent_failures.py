"""Sect. 6.1: the rewrites that all four NLI-based checkers get wrong.

The checkers are the two metrics, zero-shot and fine-tuned. The script lists the rewrites they all misclassify and
counts valid ones rejected and invalid ones accepted. Grouping them (reigns, dynasties, events, named periods) was
done by reading the list. Output: `results/statistics/persistent_failures.json`."""
from __future__ import annotations

from tempolib import setup_console
from tempolib.evaluation import STATISTICS_DIR, THRESHOLD, expert_sha, load_expert_pairs, load_results, write_json

CHECKERS = ["mdeb_zs", "rob_zs", "mdeb_ft", "rob_ft"]


def main() -> int:
    setup_console()
    rows, _ = load_expert_pairs("en")
    store = load_results("probabilities/en.json")
    assert store["expert_sha"] == expert_sha(), "results/probabilities/en.json does not match the current expert set"
    probs = {k: store["probs"]["expert"][k] for k in CHECKERS}

    failures = []
    for i, r in enumerate(rows):
        if all(int(probs[k][i] >= THRESHOLD) != r["label"] for k in CHECKERS):
            failures.append({"id": r["id"], "type": r["error_type"], "label": r["label"], "document": r["document"],
                             "claim": r["claim"], "p_faithful": {k: round(probs[k][i], 3) for k in CHECKERS}})
    rejected_valid = sum(f["label"] == 1 for f in failures)
    accepted_invalid = len(failures) - rejected_valid
    print(f"{len(failures)} of {len(rows)} rewrites ({100 * len(failures) / len(rows):.0f}%) misclassified by the "
          f"four checkers: {rejected_valid} valid rejected, {accepted_invalid} invalid accepted\n")
    for f in failures:
        print(f"  {'valid  ' if f['label'] else 'invalid'} {f['type']:14} {f['claim']}")
    write_json(STATISTICS_DIR / "persistent_failures.json",
               {"checkers": CHECKERS, "n_pairs": len(rows), "misclassified": len(failures),
                "valid_rejected": rejected_valid, "invalid_accepted": accepted_invalid, "rewrites": failures},
               indent=1)
    print("\n-> results/statistics/persistent_failures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
