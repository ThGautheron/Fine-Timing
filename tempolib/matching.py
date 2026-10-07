"""Match a summary sentence with the retrieved source passages, by embeddings or by shared words."""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import Callable, Optional

from .temporal import Expression, get_window

STOPWORDS = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "a", "au", "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce", "cet", "cette", "ces", "qui", "que", "dont", "est", "sont", "ete", "etre", "avoir", "il", "elle", "ils", "elles", "on", "se", "sa", "son", "ses", "leur", "leurs", "plus", "moins", "tres", "bien", "aussi", "donc", "mais", "car", "ne", "pas", "y", "d", "l", "s", "n", "c", "qu", "j", "m", "t"}

EMBEDDING_MODEL = "BAAI/bge-m3"

_state = {"lexical_only": False}
_model_cache: list = []


def set_lexical_only(flag: bool) -> None:
    _state["lexical_only"] = bool(flag)


def lexical_sim(a: str, b: str) -> float:
    ta = {w for w in re.findall(r"\b[a-zàâäéèêëîïôöùûüç]{3,}\b", a.lower()) if w not in STOPWORDS}
    tb = {w for w in re.findall(r"\b[a-zàâäéèêëîïôöùûüç]{3,}\b", b.lower()) if w not in STOPWORDS}
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def load_model():
    if _state["lexical_only"]:
        return None
    if not _model_cache:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _model_cache.append(SentenceTransformer(EMBEDDING_MODEL, device=dev))
        except Exception as e:
            print(f"! {EMBEDDING_MODEL} unavailable ({e}), falling back to lexical matching.", file=sys.stderr)
            _model_cache.append(None)
    return _model_cache[0]


@lru_cache(maxsize=4096)
def encode(text):
    m = load_model()
    return m.encode(text, normalize_embeddings=True) if m else None


def best_match(query: str, candidates: list[str]) -> tuple[Optional[str], float]:
    if not candidates:
        return None, 0.0
    embs = encode(tuple(candidates))
    q_emb = encode(query) if embs is not None else None

    if embs is not None and q_emb is not None:
        scores = embs @ q_emb
        idx = int(scores.argmax())
        return candidates[idx], float(scores[idx])

    return max(((c, lexical_sim(query, c)) for c in candidates), key=lambda x: x[1])


def locate_in_chunk(sentence: str, expr: Expression, chunk: dict, segment: Callable[[str], list[str]]) -> Optional[str]:
    ctx = chunk.get("context", "")
    candidates = segment(ctx) or [ctx]
    if not candidates:
        return None
    query = get_window(sentence, expr)
    if load_model() is None and not lexical_sim(query, query):
        query = sentence
    return best_match(query, candidates)[0]
