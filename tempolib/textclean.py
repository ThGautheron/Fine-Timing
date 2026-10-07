"""Normalise claim and document text."""
from __future__ import annotations

import re
from typing import Callable

from . import extraction_fr
from .dedup import normalize

_CITE_CLOSED = re.compile(r"\s*\[[^\[\]]*\]")
_CITE_OPEN_TAIL = re.compile(r"\s*\[[^\[\]]*$")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_DANGLING_OPEN_PAREN = re.compile(r"\s*\([^()]*$")
_TRAILING_PUNCT = re.compile(r"\s*[:;,\-–—(]+\s*$")
_LEADING_LIST = re.compile(r"^[\s>#*\-–—•]+")

WINDOW_TOKENIZER = "microsoft/mdeberta-v3-base"
_tok_cache: list = []


def clean_claim(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = _CITE_CLOSED.sub("", t)
    t = _CITE_OPEN_TAIL.sub("", t)
    if t.count("(") < t.count(")"):
        t = t.replace(")", "", t.count(")") - t.count("("))
    elif t.count("(") > t.count(")"):
        t = _DANGLING_OPEN_PAREN.sub("", t)
    t = _LEADING_LIST.sub("", t)
    t = _MULTISPACE.sub(" ", t).strip()
    t = _TRAILING_PUNCT.sub("", t).strip()
    return t


def _tokenizer():
    if not _tok_cache:
        from transformers import AutoTokenizer

        _tok_cache.append(AutoTokenizer.from_pretrained(WINDOW_TOKENIZER, use_fast=False))
    return _tok_cache[0]


def _ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def window_document(document: str, focus: str, max_tokens: int = 850,
                    segment: Callable[[str], list[str]] = extraction_fr.segment) -> str:
    """Shorten a document to about `max_tokens` tokens, around the sentence it should support."""
    doc = (document or "").strip()
    if max_tokens <= 0 or not doc:
        return doc
    tok = _tokenizer()
    if _ntok(tok, doc) <= max_tokens:
        return doc

    sents = segment(doc) or [doc]
    if len(sents) == 1:
        ids = tok(doc, add_special_tokens=False)["input_ids"][:max_tokens]
        return tok.decode(ids).strip()

    f = normalize(focus or "")
    fi, best = 0, -1.0
    for i, s in enumerate(sents):
        ns = normalize(s)
        score = 1.0 if f and f in ns else _overlap(f, ns)
        if score > best:
            best, fi = score, i

    lens = [_ntok(tok, s) for s in sents]
    lo = hi = fi
    total = lens[fi]
    while True:
        cand = []
        if lo > 0:
            cand.append((fi - (lo - 1), lo - 1))
        if hi < len(sents) - 1:
            cand.append((hi + 1 - fi, hi + 1))
        if not cand:
            break
        _, j = min(cand)
        if total + lens[j] > max_tokens:
            break
        total += lens[j]
        if j < lo:
            lo = j
        else:
            hi = j
    window = " ".join(sents[lo:hi + 1])
    ids = tok(window, add_special_tokens=False)["input_ids"]
    if len(ids) > max_tokens:
        window = tok.decode(ids[:max_tokens]).strip()
    return window
