"""Find temporal expressions in English text and turn them into intervals."""
from __future__ import annotations

import re
from typing import Optional

from .temporal import (
    MAX_YR, MIN_YR, Expression, Normalization, century_to_interval, clean_md, compare_intervals,
    compatible_registries,
)

__all__ = ["NOISE", "all_intervals", "anchor_expr", "clean_md", "extract", "first_interval", "is_bib",
           "is_noise", "segment"]

_YR = r"(?:1[0-9]{3}|20[0-2][0-9])"
_YR_G = rf"({_YR})"

_WORD_ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty-first": 21, "twenty first": 21,
}
_ORD_ALT = "|".join(sorted((re.escape(w) for w in _WORD_ORD), key=len, reverse=True))
_ORD = rf"(?:\d{{1,2}}(?:st|nd|rd|th)|{_ORD_ALT})"


def _ord_to_int(tok: Optional[str]) -> Optional[int]:
    if not tok:
        return None
    t = re.sub(r"\s+", " ", tok.strip().lower())
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", t)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 30 else None
    return _WORD_ORD.get(t)


def _year_or_century(s: str) -> Optional[int]:
    s = s.strip()
    if re.fullmatch(_YR, s):
        return int(s)
    m = re.search(rf"({_ORD})[\s\-]+centur(?:y|ies)", s, re.I)
    if m:
        n = _ord_to_int(m.group(1))
        if n:
            return century_to_interval(n)[0]
    return None


_PERIODS_SRC = [
    ("first millennium bc", r"first\s+millennium\s+(?:bc|bce)", (-1000, -1)),
    ("second millennium bc", r"second\s+millennium\s+(?:bc|bce)", (-2000, -1001)),
    ("first millennium", r"first\s+millennium(?:\s+(?:ad|ce))?", (1, 1000)),
    ("second millennium", r"second\s+millennium(?:\s+(?:ad|ce))?", (1001, 2000)),
    ("third millennium", r"third\s+millennium(?:\s+(?:ad|ce))?", (2001, 3000)),
    ("long nineteenth century", r"long\s+nineteenth\s+century", (1789, 1914)),
    ("early middle ages", r"early\s+middle\s+ages|early\s+medieval\s+(?:period|era|centuries)", (500, 1000)),
    ("high middle ages", r"high\s+middle\s+ages", (1000, 1300)),
    ("central middle ages", r"central\s+middle\s+ages", (1000, 1300)),
    ("late middle ages", r"late\s+middle\s+ages|late\s+medieval\s+(?:period|era)", (1300, 1500)),
    ("middle ages", r"middle\s+ages|medieval\s+(?:period|era|times|age)", (500, 1500)),
    ("late antiquity", r"late\s+antiquity", (300, 700)),
    ("classical antiquity", r"classical\s+antiquity", (-800, 500)),
    ("antiquity", r"antiquity", (-800, 500)),
    ("renaissance", r"renaissance", (1400, 1600)),
    ("early modern", r"early\s+modern\s+(?:period|era|europe|world|age)", (1500, 1800)),
    ("enlightenment", r"age\s+of\s+enlightenment|enlightenment", (1650, 1800)),
    ("victorian era", r"victorian\s+(?:era|period|age)", (1837, 1901)),
    ("georgian era", r"georgian\s+(?:era|period)", (1714, 1837)),
    ("regency era", r"regency\s+(?:era|period)", (1811, 1820)),
    ("edwardian era", r"edwardian\s+(?:era|period)", (1901, 1910)),
    ("belle epoque", r"belle\s+[ée]poque", (1871, 1914)),
    ("gilded age", r"gilded\s+age", (1870, 1900)),
    ("progressive era", r"progressive\s+era", (1896, 1917)),
    ("interwar period", r"interwar\s+(?:period|era)", (1918, 1939)),
    ("romantic era", r"romantic\s+(?:era|period)", (1800, 1850)),
    ("industrial revolution", r"industrial\s+revolution", (1760, 1840)),
    ("contemporary period", r"contemporary\s+(?:period|era)", (1945, 2100)),
    ("modern period", r"modern\s+(?:period|era)", (1800, 2100)),
]
_PERIOD_BOUNDS = {k: b for k, _, b in _PERIODS_SRC}

_MONTHS = (r"January|February|March|April|May|June|July|August|September|October|"
           r"November|December")
_DASHC = r"\-‐-―−"
_DASH = rf"[{_DASHC}]"
_MOD = (r"(?:early|mid|middle|late|(?:beginning|start)\s+of(?:\s+the)?|end\s+of(?:\s+the)?|"
        r"turn\s+of(?:\s+the)?|first\s+half\s+of(?:\s+the)?|second\s+half\s+of(?:\s+the)?)")
_MODSEP = rf"[\s{_DASHC}]+"

_PAT_SRC = [
    ("iso_date",
     rf"\b{_YR_G}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b", "absolute_date"),
    ("dmy",
     rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\s+{_YR_G}\b", "absolute_date"),
    ("mdy",
     rf"\b(?:{_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YR_G}\b", "absolute_date"),
    ("interval_centuries",
     rf"\b(?:between\s+the\s+|from\s+the\s+|the\s+)?(?:{_MOD}{_MODSEP})?({_ORD})\s*{_DASH}?\s*"
     rf"(?:{_DASH}\s*|(?:to|through|and)\s+)\s*(?:the\s+)?(?:{_MOD}{_MODSEP})?({_ORD})"
     rf"{_MODSEP}centur(?:y|ies)(?:\s+(BCE?|BC|CE|AD))?\b", "interval"),
    ("interval_years",
     rf"\b(?:from\s+|between\s+)?{_YR_G}\s*(?:{_DASH}|\s+(?:to|and|through|until)\s+)\s*{_YR_G}\b",
     "interval"),
    ("month_year",
     rf"\b(?:{_MONTHS})\s+(?:of\s+)?{_YR_G}\b", "absolute_date"),
    ("season_year",
     rf"\b(?:spring|summer|autumn|fall|winter)\s+(?:of\s+)?{_YR_G}\b", "absolute_date"),
    ("circa_year",
     rf"\b(?:circa|around|ca\.|c\.)\s+{_YR_G}\b", "absolute_date"),
    ("bound_before",
     rf"\b(?:before|prior\s+to|up\s+(?:un)?til|until|earlier\s+than|no\s+later\s+than|by)\s+"
     rf"({_YR}|(?:the\s+)?(?:{_MOD}{_MODSEP})?{_ORD}{_MODSEP}centur(?:y|ies))\b", "interval"),
    ("bound_after",
     rf"\b(?:after|since|as\s+early\s+as|no\s+earlier\s+than|later\s+than|"
     rf"starting\s+(?:in|from)|beginning\s+in)\s+"
     rf"(?:year\s+)?({_YR}|(?:the\s+)?(?:{_MOD}{_MODSEP})?{_ORD}{_MODSEP}centur(?:y|ies))\b", "interval"),
    ("decade",
     rf"\b(?:the\s+)?(?:(early|mid|middle|late)[\s{_DASHC}]+)?((?:1[0-9]|20)[0-9]0)s\b", "interval"),
]
_PAT_SRC += [(f"period:{k}", rf"\b(?:({_MOD}){_MODSEP})?(?:the\s+)?(?:{rx})\b", "named_period")
             for k, rx, _ in _PERIODS_SRC]
_PAT_SRC += [
    ("century",
     rf"\b(?:the\s+)?(?:({_MOD}){_MODSEP})?({_ORD}){_MODSEP}centur(?:y|ies)(?:\s+(BCE?|BC|CE|AD))?\b",
     "named_period"),
    ("year", rf"\b{_YR_G}\b", "absolute_date"),
    ("deictic",
     r"\b(?:nowadays|currently|presently|at\s+present|these\s+days|recently|"
     r"in\s+recent\s+years|previously|formerly|hitherto|to\s+date|thus\s+far|"
     r"so\s+far|henceforth|thereafter|subsequently|since\s+then|at\s+the\s+time|"
     r"at\s+that\s+time|until\s+now|up\s+to\s+now)\b", "relative"),
]

PATTERNS = [(name, re.compile(rx, re.I), etype) for name, rx, etype in _PAT_SRC]

NOISE = [
    re.compile(r"https?://\S+|\bdoi:\s*\S+", re.I),
    re.compile(r"\b\S+\.(?:pdf|xml|html?|csv|json|txt)\b", re.I),
    re.compile(r"\[[^\]\n]{1,200}\]"),
    re.compile(r"\bpp?\.\s*\d+\s*" + _DASH + r"\s*\d+|\bvol\.\s*\d+|\bnos?\.\s*\d+\b|\bp\.\s*\d+\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b"),
    re.compile(r"\bIS[BS]N\b[\s\d" + _DASHC + r"]+", re.I),
]

_BIB_RX = re.compile(
    r"\bpp?\.\s*\d+\s*" + _DASH + r"\s*\d"
    r"|https?://|\bdoi:"
    r"|\bUniversity\s+Press\b"
    r"|\b(?:Retrieved|Accessed)\b[^.]*\b\d{4}"
    r"|\b(?:eds?|vol|nos?)\.\s*\d"
    r"|\bin\s+[A-Z][A-Za-z.'-]+\s+(?:and|&|et\s+al\.)[^.]*\b\d{4}\b"
    r"|\|\s*(?:19|20)\d\d\s*$"
    r"|^\s*\S[\w .,'&-]{0,60},\s*\d+\s*\|"
    r"|\b\d{3,}/[\w.]+/+\d",
    re.I | re.M,
)

_ABBRV = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|al|Fig|Figs|Nos?|Vols?|pp|Eds?|Rev|"
    r"Inc|Ltd|Co|Corp|Dept|Univ|ca|approx|cf|esp|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|"
    r"Oct|Nov|Dec|e\.g|i\.e|ibid|op|cit|c)\.\s*)$",
    re.I,
)
_INITIAL = re.compile(r"(?<![A-Za-z])(?:[A-Z]|[ei])\.\s*$")


def _fail() -> Normalization:
    return Normalization(is_valid=False, granularity="unknown")


def normalize_match(pattern: str, m: "re.Match") -> Normalization:
    """Turn a regex match of the named pattern into an interval of years with its granularity."""
    if pattern == "year":
        y = int(m.group(1))
        return Normalization(y, y, "year") if MIN_YR <= y <= MAX_YR else _fail()

    if pattern == "iso_date":
        y = int(m.group(1))
        return Normalization(y, y, "day") if MIN_YR <= y <= MAX_YR else _fail()

    if pattern in ("dmy", "mdy", "month_year", "season_year", "circa_year"):
        y = int(m.group(m.lastindex))
        if not MIN_YR <= y <= MAX_YR:
            return _fail()
        gran = {"dmy": "day", "mdy": "day", "month_year": "month",
                "season_year": "season", "circa_year": "year"}[pattern]
        return Normalization(y, y, gran)

    if pattern == "interval_years":
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        if not (MIN_YR <= a <= MAX_YR and MIN_YR <= b <= MAX_YR):
            return _fail()
        return Normalization(a, b, "year")

    if pattern == "decade":
        mod = (m.group(1) or "").lower()
        base = (int(m.group(2)) // 10) * 10
        lo, hi = base, base + 9
        if mod == "early":
            hi = base + 3
        elif mod in ("mid", "middle"):
            lo, hi = base + 3, base + 6
        elif mod == "late":
            lo = base + 6
        return Normalization(lo, hi, "decade")

    if pattern == "century":
        n = _ord_to_int(m.group(2))
        if not n:
            return _fail()
        era = (m.group(3) or "").upper().replace(".", "")
        if era in ("BC", "BCE"):
            return Normalization(-100 * n, -100 * (n - 1) - 1, "century")
        lo, hi = century_to_interval(n)
        mod = re.sub(r"\s+", " ", (m.group(1) or "").lower()).strip()
        if mod in ("early", "beginning of the"):
            hi = lo + 39
        elif mod in ("mid", "middle"):
            lo, hi = lo + 30, lo + 69
        elif mod in ("late", "end of the"):
            lo = lo + 60
        elif mod == "first half of the":
            hi = lo + 49
        elif mod == "second half of the":
            lo = lo + 50
        elif mod == "turn of the":
            lo, hi = lo - 15, lo + 14
        return Normalization(lo, hi, "century")

    if pattern == "interval_centuries":
        a, b = _ord_to_int(m.group(1)), _ord_to_int(m.group(2))
        if not a or not b:
            return _fail()
        a, b = sorted((a, b))
        era = (m.group(3) or "").upper().replace(".", "")
        if era in ("BC", "BCE"):
            return Normalization(-100 * b, -100 * (a - 1) - 1, "century")
        return Normalization(century_to_interval(a)[0], century_to_interval(b)[1], "century")

    if pattern.startswith("period:"):
        lo, hi = _PERIOD_BOUNDS[pattern[7:]]
        mod = re.sub(r"\s+", " ", (m.group(1) or "").lower()).strip()
        width = hi - lo + 1
        if mod.startswith(("early", "beginning of", "start of")):
            hi = lo + round(width * 0.4) - 1
        elif mod.startswith(("mid", "middle")):
            lo, hi = lo + round(width * 0.3), lo + round(width * 0.7) - 1
        elif mod.startswith(("late", "end of")):
            lo = lo + round(width * 0.6)
        elif mod.startswith("first half of"):
            hi = lo + round(width * 0.5) - 1
        elif mod.startswith("second half of"):
            lo = lo + round(width * 0.5)
        return Normalization(lo, hi, "period")

    if pattern == "bound_before":
        b = _year_or_century(m.group(1))
        return Normalization(None, b, "year") if b is not None else _fail()

    if pattern == "bound_after":
        b = _year_or_century(m.group(1))
        return Normalization(b, None, "year") if b is not None else _fail()

    return _fail()


def extract(text: str, mask_noise: bool = True) -> list[Expression]:
    target = text
    if mask_noise:
        for pat in NOISE:
            target = pat.sub(lambda mm: " " * len(mm.group(0)), target)

    found, occupied = [], []
    for name, pat, e_type in PATTERNS:
        for m in pat.finditer(target):
            s, e = m.start(), m.end()
            if any(s < oe and e > os for os, oe in occupied):
                continue
            found.append(Expression(text[s:e].strip(), s, e, e_type, normalize_match(name, m)))
            occupied.append((s, e))
    return sorted(found, key=lambda x: x.char_start)


def first_interval(text: str) -> Optional[tuple]:
    for e in extract(text):
        n = e.normalization
        if n.is_valid and n.start is not None:
            end = n.end if n.end is not None else n.start
            return ((n.start, end), n.granularity)
    return None


def all_intervals(text: str) -> list[tuple]:
    out = []
    for e in extract(text):
        n = e.normalization
        if n.is_valid and n.start is not None:
            end = n.end if n.end is not None else n.start
            out.append(((n.start, end), n.granularity, e.surface))
    return out


def anchor_expr(expr: Expression, chunk_txt: str) -> Optional[str]:
    """Relation (equal, contains, included...) between the expression and the closest date of the source chunk; 'absent' if there is none."""
    if not chunk_txt:
        return None
    c_exprs = [e for e in extract(chunk_txt) if e.normalization.is_valid]
    if not c_exprs:
        return "absent"

    n = expr.normalization
    if not n.is_valid:
        return None

    def sort_key(c: Expression):
        cn = c.normalization
        compat = compatible_registries(n.start, n.end, cn.start, cn.end)
        rel = compare_intervals(n.start, n.end, cn.start, cn.end)
        rank = {"equal": 0, "included": 1, "contains": 1, "overlaps": 2,
                "disjoint": 3, None: 4}.get(rel, 4)
        return (0 if compat else 1, rank, abs((n.start or 0) - (cn.start or 0)))

    best = min(c_exprs, key=sort_key).normalization
    if not compatible_registries(n.start, n.end, best.start, best.end):
        return "absent"
    return compare_intervals(n.start, n.end, best.start, best.end)


def segment(text: str) -> list[str]:
    phrases, curr = [], []
    for m in re.finditer(r"[^.!?;\n]*[.!?;\n]|\S+$", text):
        curr.append(m.group(0))
        joined = "".join(curr)
        js = joined.rstrip()
        if _ABBRV.search(js) or _INITIAL.search(js):
            continue
        if joined.strip():
            phrases.append(joined.strip())
        curr = []
    if curr and "".join(curr).strip():
        phrases.append("".join(curr).strip())
    return [p for p in phrases if len(p) > 15]


def is_noise(sentence: str) -> bool:
    return bool(
        re.search(r"\.(?:pdf|xml)\b|https?://|^\s*(?:references|bibliography|works cited)\b",
                  sentence or "", re.I)
        or (sentence or "").count("[") >= 3
        or (sentence or "").count("|") >= 3
        or re.fullmatch(r"[#*\s\-—>]+.{0,40}", sentence or "")
    )


def is_bib(sentence: str) -> bool:
    return bool(_BIB_RX.search(sentence)) or is_noise(sentence)
