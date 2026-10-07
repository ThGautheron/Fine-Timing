"""LLM judges on the English expert set, through any OpenAI-compatible server.

For each (document, claim) pair the model gets the criterion written as rules and answers with a verdict and a
confidence. Answers are cached in `results/llm_judge/<model>__rules.jsonl`, so a cached set needs no server. LLMs are
not deterministic: a fresh run can differ a little from the shipped one."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

from tempolib import setup_console
from tempolib.evaluation import JUDGE_DIR, evaluation_sets, expert_sha, write_json

PROMPT = """You check whether a claim is temporally faithful to a source document.
Judge ONLY the temporal information of the claim (dates, periods, time spans). Ignore any other content of the claim that the document does not mention.

Apply these rules exactly:
1. Granularity tiers, from finest to widest: day < month (or season) < year < decade < quarter/half of a century < century < named era.
2. SAME INTERVAL (rewording, other notation) -> FAITHFUL.
3. WIDER PERIOD that contains the documented date:
   - FAITHFUL only if it is at most 3 tiers wider than the document's expression.
   - UNFAITHFUL if it is wider than that.
   - A named reign, dynasty or event counts as a period with its known dates: FAITHFUL if it contains the documented date and respects the 3-tier limit.
4. NARROWER (more precise) date: FAITHFUL only if some date stated in the document falls within the claim's interval; otherwise UNFAITHFUL (invented precision).
5. A date or period that does not contain the documented date, or only overlaps it (shifted by a day, a month, a year or a century), -> UNFAITHFUL.

Examples, from real summaries ("doc" is what the document states, "claim" what the claim states):
SAME INTERVAL
- doc "the 19th century", claim "the 1800s" -> faithful
- doc "1840–1914", claim "between 1840 and 1914" -> faithful
WIDER PERIOD
- doc "1968", claim "the 1960s" -> faithful
- doc "1478", claim "the Middle Ages" -> unfaithful
NARROWER DATE
- doc "the 1960s" (and, elsewhere, "1966"), claim "the late 1960s" -> faithful
- doc "the 19th century" (no date of 1890 elsewhere), claim "1890" -> unfaithful 
DATE NOT CONTAINING THE DOCUMENTED DATE
- doc "the 1740s", claim "the 1750s" -> unfaithful
- doc "1688", claim "1689" -> unfaithful

Document:
\"\"\"
{doc}
\"\"\"

Claim: "{claim}"

Reply with JSON only: {{"verdict": "faithful" or "unfaithful", "confidence": integer from 0 to 100}}"""
THINK = re.compile(r"<think>.*?</think>", re.S)
MODELS = ["qwen/qwen3-4b-2507", "qwen/qwen3-30b-a3b-2507"]
SUFFIX = "rules"


def parse_reply(txt):
    """Read the verdict and the confidence (0 to 1) out of a model reply; (None, None) if it cannot be parsed."""
    t = THINK.sub("", txt or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    v = c = None
    if m:
        try:
            j = json.loads(m.group(0))
            v = str(j.get("verdict", "")).lower()
            c = j.get("confidence")
        except Exception:
            pass
    if v is None:
        low = t.lower()
        v = "unfaithful" if "unfaithful" in low else "faithful" if "faithful" in low else None
    if v not in ("faithful", "unfaithful"):
        return None, None
    try:
        c = max(0, min(100, float(c))) / 100
    except (TypeError, ValueError):
        c = 1.0
    return v, c


def probability(verdict, confidence):
    """Turn a verdict and its confidence into a pseudo-probability of `faithful`, on the right side of 0.5."""
    if verdict is None:
        return 0.0
    return max(0.5, confidence) if verdict == "faithful" else min(0.4999, 1 - confidence)


def item_key(salt, doc, claim):
    return hashlib.sha1(f"{salt}|{doc}|{claim}".encode()).hexdigest()[:16]


def judge_pairs(model, pairs, base_url, cache_fp, salt="", max_tokens=60):
    """Verdicts of `model` for (document, claim) pairs; the ones missing from the cache are asked to the server."""
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_fp.exists():
        for line in open(cache_fp, encoding="utf-8"):
            x = json.loads(line)
            done[x["key"]] = x
    client, n_new, t0 = None, 0, time.time()
    with open(cache_fp, "a", encoding="utf-8") as out:
        for i, (doc, claim) in enumerate(pairs):
            k = item_key(salt, doc, claim)
            if k in done:
                continue
            if client is None:
                from openai import OpenAI

                client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)
            txt = ""
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model, temperature=0, seed=42, max_tokens=max_tokens,
                        messages=[{"role": "user", "content": PROMPT.format(doc=doc, claim=claim)}])
                    txt = resp.choices[0].message.content or ""
                    break
                except Exception as e:
                    txt = f"ERROR: {e}"
                    time.sleep(5 * (attempt + 1))
            v, c = parse_reply(txt)
            done[k] = {"key": k, "i": i, "raw": txt, "verdict": v, "confidence": c}
            out.write(json.dumps(done[k], ensure_ascii=False) + "\n")
            out.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f"  {n_new} new items, {(time.time() - t0) / n_new:.1f} s/item", flush=True)
    print(f"{model}: {n_new} new verdicts -> {cache_fp}")
    return [done[item_key(salt, doc, claim)] for doc, claim in pairs]


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    args = ap.parse_args()

    sha = expert_sha()
    rows = evaluation_sets()["expert"]
    pairs = [(r["doc"], r["claim"]) for r in rows]
    for model in args.models:
        tag = re.sub(r"[^A-Za-z0-9.-]+", "_", model) + "__" + SUFFIX
        verdicts = judge_pairs(model, pairs, args.base_url, JUDGE_DIR / f"{tag}.jsonl", salt=f"expert|{sha}")
        probs = [probability(x["verdict"], x["confidence"]) for x in verdicts]
        write_json(JUDGE_DIR / f"{tag}.json",
                   {"model": model, "expert_sha": sha, "probs": {"expert": probs},
                    "unparsed": sum(x["verdict"] is None for x in verdicts)})
        print(f"-> results/llm_judge/{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
