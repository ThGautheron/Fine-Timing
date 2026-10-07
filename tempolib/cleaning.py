"""Clean source documents: mask noise, URLs and page markers, and cut the bibliography."""
from __future__ import annotations

import re

from .extraction_fr import NOISE as DEFAULT_NOISE

_URL = re.compile(r"https?://\S+")
_PAGE = re.compile(
    r"\b\d{1,3}\s+sur\s+\d{1,3}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{4},?\s*\d{1,2}[:h]\d{2}\b"
    r"|\bp{1,2}\.\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?\b",
    re.I,
)
_BIB_LINE = re.compile(r"^\s*\[[^\]\n]{1,60}\d{4}[^\]\n]*\][^\n]*$", re.M)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINL = re.compile(r"\n{3,}")

_BIB_MARKERS = (
    "\nbibliographie",
    "\nréférences citées",
    "\nreferences citees",
    "\nréférences bibliographiques",
    "\nreferences\n",
)


def clean_chunk(text: str, strip_bibliography: bool = True, noise=DEFAULT_NOISE) -> str:
    """A source chunk without noise, URLs and page markers, and optionally without its bibliography."""
    if not text:
        return ""
    t = text
    for pat in noise:
        t = pat.sub(lambda m: " " * len(m.group(0)), t)
    t = _URL.sub(" ", t)
    t = _PAGE.sub(" ", t)

    if strip_bibliography:
        low = t.lower()
        cut = min((low.find(m) for m in _BIB_MARKERS if low.find(m) != -1), default=-1)
        if cut != -1 and cut >= 0.4 * len(t):
            t = t[:cut]
        t = _BIB_LINE.sub(" ", t)

    t = _MULTISPACE.sub(" ", t)
    t = _MULTINL.sub("\n\n", t)
    return t.strip()


def ensure_contains(document: str, supporting_sentence: str) -> str:
    supporting_sentence = (supporting_sentence or "").strip()
    if not supporting_sentence or supporting_sentence in document:
        return document
    norm_doc = _MULTISPACE.sub(" ", document.replace("\n", " "))
    if _MULTISPACE.sub(" ", supporting_sentence) in norm_doc:
        return document
    return f"{supporting_sentence}\n\n{document}".strip()
