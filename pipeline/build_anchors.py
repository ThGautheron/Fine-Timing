"""Find the dated expressions of RAG summaries and pair each with a source sentence (Sect. 4.1).

For every dated expression of a summary sentence, the closest retrieved chunk is picked by semantic similarity (bge-m3),
then the compatible date of that chunk that fits best. The relation between the two intervals suggests how to
annotate it. Annotations of an earlier table can be carried over."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tempolib import extraction_en as ex
from tempolib import matching
from tempolib.anchors import ERROR_TYPE_NORM, read_anchor_table
from tempolib.temporal import compare_intervals, compatible_registries
from tempolib.utils import read_jsonl, resolve, setup_console

COLUMNS = [
    "item_id", "doc_id", "sentence", "expression", "source_window", "source_chunk", "source_file",
    "source_pub_year", "annotation", "stratum", "validity", "suggested_source_date",
    "suggested_source_origin", "suggested_relation",
]

_REL_RANK = {"equal": 0, "included": 1, "contains": 1, "overlaps": 2}
_REL_RANK_ALL = {"equal": 0, "contains": 1, "included": 1, "overlaps": 2, "disjoint": 3}
_REL_TYPE = {"equal": "none", "contains": "generalization", "included": "precision"}
_REL_VALIDITY = {"equal": "valid", "contains": "valid", "included": "valid"}
_JUSTIFY = {
    "none": {"equal"},
    "generalization": {"equal", "contains"},
    "precision": {"equal", "included"},
    "reformulation": {"equal", "included", "contains"},
}
_COARSE = {"period", "century"}
_MAX_CHUNK_RANK = 5
_CITATION = re.compile(r"\[([^\]\n]{1,200})\]")

_EMB: dict = {}


def _dated_sentences(chunks: list[dict]) -> list[tuple]:
    out = []
    for ch in chunks:
        sents = ex.segment(ch.get("context", "") or "")
        for si, s in enumerate(sents):
            if ex.is_bib(s):
                continue
            ivs = [iv for iv, _, _ in ex.all_intervals(s)]
            if ivs:
                out.append((s, ivs, ch, si, sents))
    return out


def _window(si: int, sents: list[str], radius: int = 1) -> str:
    lo, hi = max(0, si - radius), min(len(sents), si + radius + 1)
    win = [s for j, s in enumerate(sents[lo:hi], lo) if j == si or not ex.is_bib(s)]
    return " ".join(win).strip()


def _chunk_document(chunk: dict) -> str:
    src = str(chunk.get("source") or "").strip()
    src = re.sub(r"\.(pdf|txt|html?|xml)$", "", src, flags=re.I) or "source"
    year = chunk.get("pub_year") or chunk.get("pub_date") or ""
    title = str(chunk.get("title") or "").strip()
    head = f"[{src}, {year}] {title}".strip()
    body = re.sub(r"\n{3,}", "\n\n", (chunk.get("context") or "").strip())
    return f"{head}\n\n{body}".strip()


def _compat_rel(es: int, ee: int, a: int, b: int) -> str | None:
    if not compatible_registries(es, ee, a, b):
        return None
    return compare_intervals(es, ee, a, b)


def _iter_dates(text: str) -> list[tuple]:
    return [(surf, iv, g) for iv, g, surf in ex.all_intervals(text or "")]


def _ground_sentence(expr, sentence: str, dated: list[tuple]) -> tuple | None:
    e_iv = ex.first_interval(expr.surface) or ex.first_interval(sentence)
    if not e_iv:
        return None
    es, ee = e_iv[0]
    best = None
    for s, ivs, ch, si, sents in dated:
        for a, b in ivs:
            rel = _compat_rel(es, ee, a, b)
            if rel not in _REL_RANK:
                continue
            key = (_REL_RANK[rel], -matching.lexical_sim(sentence, s))
            if best is None or key < best[0]:
                best = (key, _window(si, sents), s, ch, rel)
    return None if best is None else (best[1], best[2], best[3], best[4])


def _best_semantic(sentence: str, items: list[tuple], key=lambda x: x[0]):
    if not items:
        return None
    m = matching.load_model()
    if m is not None:
        emb = m.encode([key(it) for it in items], normalize_embeddings=True)
        q = m.encode(sentence, normalize_embeddings=True)
        sims = np.asarray(emb) @ np.asarray(q)
        return items[int(sims.argmax())]
    return max(items, key=lambda it: matching.lexical_sim(sentence, key(it)))


def _justify(expr, sentence: str, error_type: str, dated: list[tuple], chunks: list[dict] | None = None) -> dict:
    """Look in the retrieved chunks for a date that supports the expression with the wanted relation (equal, wider, narrower)."""
    if chunks:
        cited = _cited_chunks(sentence, chunks)
        if cited:
            dated = [t for t in dated if any(t[2] is c for c in cited)]
    e_iv = ex.first_interval(expr.surface) or ex.first_interval(sentence)
    want = _JUSTIFY.get(error_type or "none", {"equal"})
    just = []
    if e_iv:
        es, ee = e_iv[0]
        for s, _ivs, ch, si, sents in dated:
            hits = [(rel, g) for _surf, (a, b), g in _iter_dates(s)
                    if (rel := _compat_rel(es, ee, a, b)) in want
                    and not (rel == "included" and g in _COARSE)]
            if hits:
                rel = hits[0][0]
                just.append((s, _window(si, sents), ch, rel))
    if just:
        s, doc, ch, rel = _best_semantic(sentence, just, key=lambda x: x[0])
        return {"found": True, "relation": rel, "document": doc, "sentence": s, "chunk": ch}
    near = _ground_sentence(expr, sentence, dated)
    if near is None:
        return {"found": False, "relation": None, "document": "", "sentence": "", "chunk": None}
    return {"found": False, "relation": near[3],
            "document": near[0], "sentence": near[1], "chunk": near[2]}


def _warm_embeddings(texts: list[str]) -> None:
    _EMB.clear()
    m = matching.load_model()
    if m is None:
        return
    uniq = list(dict.fromkeys(s.strip() for s in texts if s and s.strip()))
    if not uniq:
        return
    try:
        vecs = np.asarray(m.encode(uniq, normalize_embeddings=True, batch_size=64, show_progress_bar=False))
        for t, v in zip(uniq, vecs):
            _EMB[t] = v
    except Exception as e:
        print(f"! warm embeddings: {e}", file=sys.stderr)
        _EMB.clear()


def _vec(text: str):
    t = (text or "").strip()
    v = _EMB.get(t)
    return v if v is not None else matching.encode(t)


def _sem_sim(a: str, b: str) -> float:
    va, vb = _vec(a), _vec(b)
    if va is None or vb is None:
        return matching.lexical_sim(a, b)
    return float(np.asarray(va) @ np.asarray(vb))


def _interval_gap(a1: int, a2: int, b1: int, b2: int) -> int:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0


def _rank_chunks_semantic(sentence: str, chunks: list[dict]) -> list[dict]:
    """The retrieved chunks, most similar to the summary sentence first."""
    qv = _vec(sentence)
    q = np.asarray(qv) if qv is not None else None
    scored = []
    for idx, c in enumerate(chunks):
        sents = ex.segment(c.get("context", "") or "") or [c.get("context", "") or ""]
        sims = []
        for s in sents:
            sv = _vec(s) if q is not None else None
            sims.append(float(np.asarray(sv) @ q) if sv is not None else matching.lexical_sim(sentence, s))
        sims.sort(reverse=True)
        top = sims[:3] or [0.0]
        score = 0.5 * top[0] + 0.5 * (sum(top) / len(top))
        try:
            rr = float(c.get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            rr = 0.0
        scored.append((score, rr, -idx, c))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [c for *_x, c in scored]


def _norm_cite(s: str) -> str:
    s = re.sub(r",?\s*(?:1[0-9]{3}|20[0-2][0-9])\s*$", "", (s or "").strip())
    s = re.sub(r"\.(?:pdf|xml|txt|html?)$", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip().lower()


def _cited_chunks(sentence: str, chunks: list[dict]) -> list[dict]:
    cites = {_norm_cite(m) for m in _CITATION.findall(sentence or "")}
    cites.discard("")
    if not cites:
        return []
    exact = [c for c in chunks
             if _norm_cite(str(c.get("title") or "")) in cites
             or _norm_cite(str(c.get("source") or "")) in cites]
    if exact:
        return exact
    return [c for c in chunks
            if (t := _norm_cite(str(c.get("title") or "")))
            and any(len(cite) > 6 and (cite in t or t in cite) for cite in cites)]


def _suggest_from_source(expr, sentence: str, chunks: list[dict], dated: list[tuple]) -> dict | None:
    """Pair an expression with the best matching date of the retrieved chunks and suggest how to annotate it."""
    e_iv = ex.first_interval(expr.surface) or ex.first_interval(sentence)
    if not e_iv:
        return None
    es, ee = e_iv[0]

    def _pack(rel, surf, iv, origin, chunk, rank, sentence="", window="", cited=False, attested=True):
        t = _REL_TYPE.get(rel, "") if attested else ""
        v = _REL_VALIDITY.get(rel, "") if attested else ""
        return {"date": surf, "interval": iv, "relation": rel, "origin": origin,
                "sentence": sentence, "window": window, "chunk": chunk, "rank": rank,
                "type": t, "validity": v, "cited": cited, "attested": attested}

    blank = {"date": "", "interval": None, "relation": "absent", "origin": "",
             "sentence": "", "window": "", "chunk": None, "rank": -1,
             "type": "", "validity": "", "cited": False, "attested": True}
    if not chunks:
        return blank

    cited = _cited_chunks(sentence, chunks)
    pool = cited or chunks

    ranked = _rank_chunks_semantic(sentence, pool)
    chosen, chosen_rank = ranked[0], 0
    for r, c in enumerate(ranked[:_MAX_CHUNK_RANK]):
        if any(t[2] is c for t in dated):
            chosen, chosen_rank = c, r
            break

    best = None
    for s, _ivs, ch, si, sents in (t for t in dated if t[2] is chosen):
        sim = _sem_sim(sentence, s)
        for surf, (a, b), g in _iter_dates(s):
            rel = _compat_rel(es, ee, a, b)
            key = (_REL_RANK_ALL.get(rel, 4), _interval_gap(es, ee, a, b), -sim)
            if best is None or key < best[0]:
                best = (key, rel, surf, (a, b), s, _window(si, sents), g)
    if best:
        _, rel, surf, iv, s, win, gran = best
        attested = not (rel == "included" and gran in _COARSE)
        return _pack(rel, surf, iv, "sentence", chosen, chosen_rank, sentence=s, window=win,
                     cited=bool(cited), attested=attested)

    py = _pub_year(chosen)
    if py:
        rel = _compat_rel(es, ee, py, py)
        if rel in _REL_TYPE:
            return _pack(rel, f"pub_date {py}", (py, py), "pub_date", chosen, chosen_rank,
                         cited=bool(cited))

    b = dict(blank)
    b["chunk"] = chosen
    b["cited"] = bool(cited)
    return b


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _pub_year(chunk: dict | None) -> int | None:
    if not chunk:
        return None
    v = chunk.get("pub_year", chunk.get("pub_date"))
    s = str(v).strip()
    if not s[:4].isdigit():
        return None
    y = int(s[:4])
    return y if y >= 1000 else None


def _load_prior_annotations(path: Path | None) -> dict[tuple[str, str, str], tuple]:
    if not path or not path.is_file():
        return {}
    out: dict[tuple[str, str, str], tuple] = {}
    for r in read_anchor_table(path):
        ann = str(r.get("annotation") or "").strip()
        stratum = str(r.get("stratum") or "").strip()
        if not ann and not stratum:
            continue
        key = (str(r.get("doc_id", "")).strip(), _norm(r.get("sentence")), _norm(r.get("expression")))
        out[key] = (ann, stratum)
    return out


def build_items(records: list[dict], prior: dict) -> tuple[list[dict], dict]:
    """One anchor per dated expression of every summary sentence, with its source sentence and a suggested annotation."""
    items: list[dict] = []
    stats = Counter()
    stats["records"] = len(records)

    for i, rec in enumerate(records):
        stats[f"theme:{_norm(rec.get('theme')) or '(empty)'}"] += 1
        chunks = rec.get("retrieve_chunk", []) or []
        dated = _dated_sentences(chunks)

        pool = [ex.clean_md(p) for p in ex.segment(rec.get("model_answer", ""))]
        for c in chunks:
            pool += ex.segment(c.get("context", "") or "")
        _warm_embeddings(pool)

        for pi, p_raw in enumerate(ex.segment(rec.get("model_answer", ""))):
            if ex.is_noise(p_raw):
                continue
            sentence = ex.clean_md(p_raw)
            exprs = ex.extract(sentence)
            if not exprs or len(sentence) < 20:
                continue

            for expr in exprs:
                key = (f"summary_{i:02d}", _norm(sentence), _norm(expr.surface))
                carried = prior.get(key)
                if carried:
                    annotation, stratum = carried
                    stats["annotations carried over"] += 1
                else:
                    annotation, stratum = "", ""
                error_type = ERROR_TYPE_NORM.get(annotation.strip().lower(), "none")

                validity = ""
                relation = None
                date_src = date_origin = ""
                sugg = _suggest_from_source(expr, sentence, chunks, dated)
                if sugg and sugg.get("cited"):
                    stats["chunk restricted by citation [..]"] += 1
                if sugg and not sugg.get("attested", True):
                    stats["precision rejected (coarse source grain: period/century)"] += 1
                if sugg is not None:
                    relation = sugg["relation"]
                    date_src, date_origin = sugg["date"], sugg["origin"]

                if not carried:
                    if sugg is not None:
                        if sugg["type"]:
                            annotation = sugg["type"]
                        if sugg["validity"]:
                            validity = sugg["validity"]
                        error_type = ERROR_TYPE_NORM.get(annotation.strip().lower(), "none")
                        stats[f"suggested: type={annotation or '(empty)'}"] += 1
                        stats[f"suggested: source={date_origin or '(none)'}"] += 1
                        stats[f"suggested: chunk rank {sugg['rank'] if sugg['rank'] >= 0 else 'n/a'}"] += 1
                elif error_type in ("generalization", "precision"):
                    j = _justify(expr, sentence, error_type, dated, chunks)
                    validity = "valid" if j["found"] else "invalid"
                    relation = j["relation"]
                    stats[f"{error_type} {'justified' if j['found'] else 'NOT justified'}"] += 1

                chunk = sugg["chunk"] if sugg and sugg["chunk"] else None
                if not chunk:
                    stats["sentence without chunk -> skipped"] += 1
                    continue
                window = ((sugg["window"] if sugg and sugg["window"] else "")
                          or (sugg["sentence"] if sugg and sugg["sentence"] else "")
                          or matching.locate_in_chunk(sentence, expr, chunk, ex.segment) or "")
                full_chunk = _chunk_document(chunk)
                pub_year = _pub_year(chunk)

                items.append({
                    "item_id": f"sp_{i:02d}_{pi:03d}_"
                               f"{hashlib.sha1(f'{expr.surface}||{expr.char_start}'.encode()).hexdigest()[:8]}",
                    "doc_id": f"summary_{i:02d}",
                    "sentence": sentence,
                    "expression": expr.surface,
                    "source_window": window,
                    "source_chunk": full_chunk or window,
                    "source_file": chunk.get("source") if chunk else None,
                    "source_pub_year": pub_year,
                    "annotation": annotation,
                    "stratum": stratum,
                    "validity": validity,
                    "suggested_source_date": date_src,
                    "suggested_source_origin": date_origin,
                    "suggested_relation": relation or (ex.anchor_expr(expr, window) if window else None),
                })

    stats["items"] = len(items)
    return items, dict(stats)


def _report(items: list[dict], stats: dict) -> str:
    """A short text summary of the anchors that were built."""
    by_doc = Counter(it["doc_id"] for it in items)
    pub_ok = sum(1 for it in items if it["source_pub_year"] is not None)
    L = ["=" * 70, "ANCHOR TABLE (RAG output -> anchors to annotate)", "=" * 70, ""]
    for k in ("records", "items", "sentence without chunk -> skipped", "annotations carried over",
              "generalization justified", "generalization NOT justified", "precision justified",
              "precision NOT justified"):
        if k in stats:
            L.append(f"  {k:40} {stats[k]}")
    L.append(f"  {'source_pub_year filled':32} {pub_ok}/{len(items)}")
    L += ["", "  themes:"]
    for k, v in sorted(stats.items()):
        if k.startswith("theme:"):
            L.append(f"    {k[6:]:20} {v}")
    L += ["", f"  documents: {len(by_doc)}  (min {min(by_doc.values())} / max {max(by_doc.values())} items per document)"]
    n_ann = sum(1 for it in items if it["annotation"])
    L += ["", f"  to annotate: {len(items) - n_ann} empty annotations / {len(items)}"]

    te = Counter(it["annotation"] or "(empty)" for it in items)
    va = Counter(it["validity"] or "(empty)" for it in items)
    og = Counter(it["suggested_source_origin"] or "(none)" for it in items)
    pending = sum(1 for it in items if not it["annotation"]
                  and it["suggested_relation"] in ("overlaps", "disjoint"))
    no_src = sum(1 for it in items if it["suggested_relation"] in (None, "", "absent"))
    rk = {k[len("suggested: chunk rank "):]: v for k, v in stats.items()
          if k.startswith("suggested: chunk rank ")}
    L += ["", "  suggested labels:"]
    L += ["    annotation : " + ", ".join(f"{k} {v}" for k, v in te.most_common())]
    L += ["    validity   : " + ", ".join(f"{k} {v}" for k, v in va.most_common())]
    L += ["    source date: " + ", ".join(f"{k} {v}" for k, v in og.most_common())]
    if rk:
        L += ["    chunk kept (semantic rank): " + ", ".join(f"rank {k}: {v}" for k, v in sorted(rk.items()))]
    L += [f"    overlaps/disjoint candidates to decide: {pending}"]
    L += [f"    without compatible source date       : {no_src}"]
    return "\n".join(L) + "\n"


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True, help="RAG output (JSON lines)")
    ap.add_argument("--name", required=True, help="output basename (e.g. dhq)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/anchors"))
    ap.add_argument("--carry-annotations", type=Path, default=None,
                    help="existing anchor table whose human annotations are carried over")
    ap.add_argument("--lexical", action="store_true", help="lexical matching only (no embedding model)")
    args = ap.parse_args()

    inp = resolve(args.input)
    if not inp.is_file():
        print(f"not found: {inp}")
        return 1

    matching.set_lexical_only(args.lexical)
    if not args.lexical:
        m = matching.load_model()
        print(f"{matching.EMBEDDING_MODEL}: {'unavailable (lexical fallback)' if m is None else getattr(m, 'device', '?')}")

    records = list(read_jsonl(inp))
    carry = resolve(args.carry_annotations) if args.carry_annotations else None
    prior = _load_prior_annotations(carry)
    if prior:
        print(f"{len(prior)} human annotations available to carry over ({carry.name})")

    items, stats = build_items(records, prior)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(items)

    rep = _report(items, stats)
    (out_dir / f"{args.name}_report.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep)
    print(f"-> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
