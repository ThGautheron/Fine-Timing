"""Find temporal expressions in French text; also the default sentence splitter and noise filter."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .temporal import (
    MAX_YR, MIN_YR, Expression, Normalization, century_to_interval,
)

ROMANS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
          "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18,
          "XIX": 19, "XX": 20, "XXI": 21}

PERIODS = {
    "haut moyen age": (500, 1000), "moyen age central": (1000, 1300), "moyen age classique": (1000, 1300),
    "bas moyen age": (1300, 1500), "moyen age tardif": (1300, 1500), "moyen age": (500, 1500),
    "antiquite": (-800, 500), "renaissance": (1400, 1600), "epoque moderne": (1500, 1789),
    "epoque contemporaine": (1789, 2100),
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s.lower())).strip()


def parse_roman(s: str) -> Optional[int]:
    return ROMANS.get(strip_accents(s).upper().strip())


_MONTH = r"janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre"
_SEASON = r"hiver|printemps|[ée]t[ée]|automne"
_YR_OPT = r"(?:de\s+l['’]an(?:n[ée]e)?\s+)?"
_YR_RX = r"(1[0-9]{3}|20[0-2][0-9])"

PATTERNS = [
    ("interval_years", re.compile(rf"\b(?:de\s+|entre\s+|des\s+)?{_YR_RX}\s*(?:[-‐-―]|\s+(?:a|à|et)\s+)\s*{_YR_RX}\b", re.I), "interval"),
    ("interval_centuries", re.compile(r"\b(?:(?:du|entre)\s+)?(?:le\s+)?([IVXivx]+)\s*(?:e|er|eme|ème)?\s*(?:si[eè]cle\s*)?(?:[-‐-―]|\s+(?:au|et)\s+)(?:le\s+)?\s*([IVXivx]+)\s*(?:e|er|eme|ème)?\s*si[eè]cle", re.I), "interval"),
    ("half_century", re.compile(r"\b(premi[eè]re|seconde|deuxi[eè]me|derni[eè]re)\s+moiti[ée]\s+du\s+([IVXivx]+)\s*(?:e|er|eme|ème)?\s*si[eè]cle", re.I), "named_period"),
    ("bound_before", re.compile(r"\b(?:ant[eé]rieur(?:e|es|s)?\s+(?:a|à|au)|avant|jusqu'(?:a|à|au))\s+(?:l'an\s+)?((?:1[0-9]{3}|20[0-2][0-9])|[IVXivx]+\s*(?:e|er|eme|ème)?\s*si[eè]cle)", re.I), "interval"),
    ("bound_after", re.compile(r"\b(?:post[eé]rieur(?:e|es|s)?\s+(?:a|à|au)|apr[eè]s|(?:a|à)\s+partir\s+de)\s+(?:l'an\s+)?((?:1[0-9]{3}|20[0-2][0-9])|[IVXivx]+\s*(?:e|er|eme|ème)?\s*si[eè]cle)", re.I), "interval"),
    ("decade", re.compile(rf"\b(?:les\s+|des\s+|dans\s+les\s+)?ann[eé]es\s+{_YR_RX}\b", re.I), "interval"),
    ("named_period", re.compile(r"\b(haut\s+moyen\s+[aâà]ge|bas\s+moyen\s+[aâà]ge|moyen\s+[aâà]ge\s+central|moyen\s+[aâà]ge\s+classique|moyen\s+[aâà]ge\s+tardif|moyen\s+[aâà]ge|antiquit[eé]|renaissance|[eé]poque\s+moderne|[eé]poque\s+contemporaine)\b", re.I), "named_period"),
    ("century", re.compile(r"[IVXivx]+\s*(?:e|er|eme|ème)?\s*si[eè]cle", re.I), "named_period"),
    ("day_month_year", re.compile(rf"\b(\d{{1,2}})(?:er)?\s+(?:de\s+)?(?:{_MONTH})\s+{_YR_OPT}{_YR_RX}\b", re.I), "absolute_date"),
    ("iso_date", re.compile(rf"\b{_YR_RX}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b"), "absolute_date"),
    ("month_year", re.compile(rf"\b(?:{_MONTH})\s+{_YR_OPT}{_YR_RX}\b", re.I), "absolute_date"),
    ("season_year", re.compile(rf"\b(?:{_SEASON})\s+{_YR_OPT}{_YR_RX}\b", re.I), "absolute_date"),
    ("year", re.compile(rf"\b{_YR_RX}\b"), "absolute_date"),
    ("deictic", re.compile(r"\b(d[eé]sormais|aujourd'hui|auparavant|nagu[eè]re|jadis|actuellement|(?:a|à)\s+pr[eé]sent|maintenant|r[eé]cemment|(?:a|à)\s+cette\s+[eé]poque|(?:a|à)\s+l'[eé]poque|depuis\s+lors|d[eè]s\s+lors|ant[eé]rieurement|ult[eé]rieurement|par\s+la\s+suite)\b", re.I), "relative"),
]

NOISE = [
    re.compile(r"medio_[0-9\-_a-z]+\.pdf|\bmedio_[0-9\-_]+|https?://\S+|\[source\s*:[^\]]*\]|\[[^\]]*\.pdf[^\]]*\]", re.I),
    re.compile(r"\b\d{4}-\d{4}\b(?=_)|n[o°]\s*\d{1,4}\b", re.I),
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}(?:,?\s*\d{1,2}:\d{2})?|\b\d{1,2}\s+sur\s+\d{1,2}\b"),
    re.compile(r"\b(?:U\.?A\.?|E\.?R\.?A\.?|UMR|URA|CNRS|ERA)\s*\.?\s*\d{3,4}\b|\bp{1,2}\.\s*\d{1,4}(?:\s*[-‐-―]\s*\d{1,4})?\b", re.I),
]


def normalize_match(pattern: str, m: re.Match) -> Normalization:
    """Turn a regex match of the named pattern into an interval of years with its granularity (French)."""
    fail = Normalization(is_valid=False, granularity="unknown")

    if pattern in ("year", "iso_date", "month_year", "season_year", "day_month_year"):
        y = int(m.group(1 if pattern != "day_month_year" else 2))
        if not (MIN_YR <= y <= MAX_YR):
            return fail
        gran = {"year": "year", "iso_date": "day", "day_month_year": "day", "month_year": "month",
                "season_year": "season"}
        return Normalization(y, y, gran[pattern])

    if pattern == "decade":
        base = (int(m.group(1)) // 10) * 10
        return Normalization(base, base + 9, "decade")

    if pattern == "interval_years":
        a, b = sorted([int(m.group(1)), int(m.group(2))])
        if not (MIN_YR <= a <= MAX_YR and MIN_YR <= b <= MAX_YR):
            return fail
        return Normalization(a, b, "year")

    if pattern == "interval_centuries":
        s1, s2 = parse_roman(m.group(1)), parse_roman(m.group(2))
        if not s1 or not s2:
            return fail
        s1, s2 = sorted([s1, s2])
        return Normalization(century_to_interval(s1)[0], century_to_interval(s2)[1], "century")

    if pattern == "century":
        rom = re.match(r"([IVXivx]+)", m.group(0).strip())
        s = parse_roman(rom.group(1)) if rom else None
        return Normalization(*century_to_interval(s), "century") if s else fail

    if pattern == "half_century":
        half = strip_accents(m.group(1).lower())
        s = parse_roman(m.group(2))
        if not s:
            return fail
        start, end = century_to_interval(s)
        mid = start + 49
        if half.startswith("premi"):
            return Normalization(start, mid, "half-century")
        return Normalization(mid + 1, end, "half-century")

    if pattern in ("bound_before", "bound_after"):
        val = m.group(1).strip()
        if re.fullmatch(r"1[0-9]{3}|20[0-2][0-9]", val):
            bound = int(val)
        else:
            rom = re.match(r"([IVXivx]+)", val)
            s = parse_roman(rom.group(1)) if rom else None
            if not s:
                return fail
            bound = century_to_interval(s)[0] if pattern == "bound_before" else century_to_interval(s)[1]
        return Normalization(None, bound, "year") if pattern == "bound_before" else Normalization(bound, None, "year")

    if pattern == "named_period":
        bounds = PERIODS.get(norm_key(m.group(1)))
        return Normalization(bounds[0], bounds[1], "period") if bounds else fail

    return fail


def extract(text: str, mask_noise: bool = True) -> list[Expression]:
    target = text
    if mask_noise:
        for pat in NOISE:
            target = pat.sub(lambda m: " " * len(m.group(0)), target)

    found, occupied = [], []
    for name, pat, e_type in PATTERNS:
        for m in pat.finditer(target):
            s, e = m.start(), m.end()
            if not any(s < oe and e > os for os, oe in occupied):
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


_ABBRV_RX = re.compile(r"\b(?:cf|ex|p|pp|vol|no|n°|fig|tabl|ibid|op|cit|etc|av|ap|J)\.\s*$", re.I)


def segment(text: str) -> list[str]:
    phrases, curr = [], []
    for m in re.finditer(r"[^.!?;\n]*[.!?;\n]|\S+$", text):
        curr.append(m.group(0))
        joined = "".join(curr)
        if _ABBRV_RX.search(joined.rstrip()):
            continue
        if joined.strip():
            phrases.append(joined.strip())
        curr = []
    if curr and "".join(curr).strip():
        phrases.append("".join(curr).strip())
    return [p for p in phrases if len(p) > 15]
