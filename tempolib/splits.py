"""Search for a document-level train/dev/test partition that keeps the types balanced."""
from __future__ import annotations

import random
from collections import Counter

TYPE_ORDER = ["wrong_date", "generalization", "none", "precision", "reformulation"]


def ordered_types(present) -> list[str]:
    present = set(present)
    return [t for t in TYPE_ORDER if t in present] + sorted(present - set(TYPE_ORDER))


def partition_score(test_docs: set, by_doc: dict, types: list[str], target_ratio: float, total: int) -> float:
    """How bad a candidate held-out set is (lower is better): wrong size, missing types, uneven types."""
    n_test = sum(len(by_doc[d]) for d in test_docs)
    if n_test == 0:
        return 1e9

    ratio_gap = abs(n_test / total - target_ratio)

    cnt = Counter()
    for d in test_docs:
        for row in by_doc[d]:
            cnt[row["error_type"]] += 1

    missing = sum(1 for t in types if cnt[t] == 0)

    props = [cnt[t] / n_test for t in types]
    mean = sum(props) / len(props)
    var = sum((p - mean) ** 2 for p in props) / len(props)

    return ratio_gap * 10 + missing * 5 + var * 3


def search_partition(by_doc: dict, types: list[str], ratio: float, total: int,
                     rng: random.Random, n_trials: int = 20000) -> set:
    """Draw many random sets of documents and keep the best one, about `ratio` of the anchors."""
    docs = list(by_doc)
    best, best_score = None, float("inf")

    for _ in range(n_trials):
        k = rng.randint(max(1, int(len(docs) * ratio * 0.6)),
                        max(2, int(len(docs) * ratio * 1.5)))
        cand = set(rng.sample(docs, min(k, len(docs) - 1)))
        s = partition_score(cand, by_doc, types, ratio, total)
        if s < best_score:
            best, best_score = cand, s

    return best
