"""Metric shared by the analysis scripts."""
from __future__ import annotations


def auc_ranks(p, y) -> float:
    """Area under the ROC curve, computed from average ranks (ties count one half)."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y)
    nneg = len(y) - npos
    return (sum(r for r, t in zip(ranks, y) if t) - npos * (npos + 1) / 2) / (npos * nneg)
