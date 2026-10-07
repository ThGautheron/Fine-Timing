"""The criterion: label the rewritten date of a claim as valid, tolerated or invalid.

The rewrite is compared with its anchor, the original expression with its interval and granularity. The same interval
is faithful, and so is a widening of at most three levels. A more precise date is faithful only if the source states it.
A shifted date or an over-broad period is not."""
import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from . import tiers, tiers_fr
from .tiers import ERAS, TIER_INDEX, tier_from_width

ERA_CONTAINER = re.compile(r"\b(antiquity|the middle ages|medieval period|early modern (period|era)|"
                           r"modern (period|era)|contemporary period|ancien r|roman empire|"
                           r"(first|second|third) millennium|prehistor)\b", re.I)

_POSITION = (r"(?:(?:early|late|mid|middle of|middle|beginning of|end of|turn of|junction of|"
             r"start of|dawn of|twilight of|first half of|second half of|first third of|"
             r"second third of|last third of|first quarter of|second quarter of|third quarter of|"
             r"last quarter of|(?:first|earliest|opening|early|last|final|closing|late)\s+"
             r"(?:years|decades)\s+of|opening|closing|final)(?:\s+the)?[\s-]*)?")
_ORDINAL = (r"(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|"
            r"ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
            r"seventeenth|eighteenth|nineteenth|twentieth|twenty-first)")
_ERA_SUFFIX = (r"(?:\s*(?:BCE|BC|B\.C\.|CE|AD|A\.D\.)(?!\w)"
               r"|\s+(?:before\s+(?:our\s+era|christ)|of\s+our\s+era))?")
_CONNECTOR = r"(?:\s+(?:to|and|until|through)\s+|\s*[-–—]\s*)"
_DECADE = r"(?:the\s+)?(?:early|late|mid)?[\s-]*\d{2,4}0s"
_MONTH_RE = (r"(?:january|february|march|april|may|june|july|august|september|october|"
             r"november|december)")

SPAN = re.compile(
    r"((?:(?:from|between)\s+)?(?:the\s+)?" + _POSITION + _ORDINAL + r"(?:[\s-]+centur(?:y|ies))?"
    + _ERA_SUFFIX + _CONNECTOR + r"(?:the\s+)?" + _POSITION + _ORDINAL + r"[\s-]+centur(?:y|ies)" + _ERA_SUFFIX
    + r"|(?:(?:from|between)\s+)?" + _DECADE + _CONNECTOR + _DECADE + _ERA_SUFFIX
    + r"|\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    + r"|between\s+\d{3,4}\s+and\s+\d{2,4}(?!\d)" + _ERA_SUFFIX
    + r"|(?:from\s+)?\d{3,4}\s*(?:to|until|through|-|–|—)\s*\d{2,4}(?![-\d])" + _ERA_SUFFIX
    + r"|(?:the\s+)?" + _POSITION + _ORDINAL + r"[\s-]+centur(?:y|ies)" + _ERA_SUFFIX
    + r"|" + _DECADE + _ERA_SUFFIX
    + r"|(?:the\s+)?(?:early|late|mid)?[\s-]*\d{3,4}s"
    + r"|(?:" + _MONTH_RE + r"|spring|summer|autumn|fall|winter)\s+(?:of\s+)?\d{3,4}"
    + r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE + r"\s+\d{3,4}"
    + r"|" + _MONTH_RE + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{3,4}"
    + r"|antiquity|the middle ages|medieval period|early modern period|early modern era|"
    r"modern period|modern era|contemporary period|renaissance|enlightenment|"
    r"(?:first|second|third) millennium"
    + r"|(?:AD|A\.D\.)\s+\d{1,4}|\d{1,4}\s*(?:BCE|BC|B\.C\.|CE|AD|A\.D\.)(?!\w)"
    + r"|\d{3,4}" + _ERA_SUFFIX + r")", re.I)

_ERA_MODIFIER = (r"(?:the\s+)?(?:(?:early|late|mid|middle of|middle|beginning of|end of|start of|dawn of|"
                 r"twilight of|first half of|second half of|(?:first|earliest|opening|early|last|final|"
                 r"closing|late)\s+(?:years|decades)\s+of)(?:\s+the)?)?\s*")
_ERA_NAMES = sorted({lab[4:] if lab.startswith("the ") else lab for lab in ERAS},
                    key=len, reverse=True)
ERA_SPAN = re.compile(_ERA_MODIFIER + r"(?:" + "|".join(re.escape(n) for n in _ERA_NAMES) + r")", re.I)

TIME_OF_DAY = re.compile(r"\b(morning|afternoon|evening|night|noon|midday|midnight)\s+of\s+(?:the\s+)?$", re.I)
OPEN_ANCHOR = re.compile(r"^\s*(?:by|until|till|since|before|after|as early as|as late as|"
                         r"up to|prior to)\b|^\s*from\b(?!.*(?:\bto\b|\buntil\b|\btill\b|"
                         r"\bthrough\b|[-–—]))", re.I)
HUNDREDS = re.compile(r"\b(?:1?\d)00s\b(?!\s+decade)", re.I)
TEMPORAL_HINT = re.compile(r"\d{3,4}|centur|decade|reign|dynasty|era\b|age\b|period\b|millennium", re.I)
MALFORMED_DECADE = re.compile(r"\b(\d{3,4})([1-9])s\b", re.I)
YEAR_TOKEN = re.compile(r"(?<![,.\d$£€])\b(\d{3,4})\b(?!\s*%|[.,]\d|\s*(?:BCE|BC|B\.C\.)(?!\w))")

_GRANULARITY_TIER = {"day": "day", "month": "month", "season": "month", "year": "year",
                     "decade": "decade", "century": "century", "period": "period"}


def _spans_en(text):
    out = [m.group(0).strip(" ,.;:-") for m in SPAN.finditer(text or "")]
    out += [m.group(0).strip(" ,.;:-") for m in ERA_SPAN.finditer(text or "")]
    return out


@dataclass(frozen=True)
class Language:

    name: str
    low: Callable
    to_en: Callable
    spans: Callable
    parse: Callable
    open_anchor: object
    time_of_day: object
    temporal_hint: object
    year_token: object


EN = Language("en", str.lower, lambda s: s, _spans_en, tiers.parse, OPEN_ANCHOR, TIME_OF_DAY,
              TEMPORAL_HINT, YEAR_TOKEN)
FR = Language("fr", tiers_fr.fold, lambda s: tiers_fr.to_en(s) or "", tiers_fr.spans, tiers_fr.parse,
              tiers_fr.OPEN_ANCHOR, tiers_fr.TIME_OF_DAY, tiers_fr.TEMPORAL_HINT, tiers_fr.YEAR_TOKEN)
LANGUAGES = {"en": EN, "fr": FR}


def get_language(lang) -> Language:
    if isinstance(lang, Language):
        return lang
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language: {lang!r}")
    return LANGUAGES[lang]


def claim_expression(sentence, claim, lang=EN):
    """The temporal expression the claim introduces compared with the original sentence."""
    lang = get_language(lang)
    spans_claim = lang.spans(claim)
    known = [lang.low(s) for s in lang.spans(sentence)]
    new = [s for s in spans_claim if lang.low(s) not in known]
    if new:
        return max(new, key=len)
    if spans_claim:
        return max(spans_claim, key=len)
    a, b = sentence.split(), claim.split()
    sm = difflib.SequenceMatcher(None, a, b)
    inserted = [" ".join(b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes()
                if tag in ("replace", "insert")]
    inserted = [s for s in inserted if lang.temporal_hint.search(s)]
    return max(inserted, key=len) if inserted else None


def relation(anchor, claim):
    """How the claim interval sits relative to the anchor interval: equal, contains, included, overlaps or disjoint."""
    if anchor == claim:
        return "equal"
    if claim[0] <= anchor[0] and anchor[1] <= claim[1]:
        return "contains"
    if anchor[0] <= claim[0] and claim[1] <= anchor[1]:
        return "included"
    if claim[1] < anchor[0] or anchor[1] < claim[0]:
        return "disjoint"
    return "overlaps"


def _norm_expr(s, lang):
    return re.sub(r"^(?:the|le|la|les|l'|du|des|au|aux|en)\s*", "",
                  " ".join(lang.low(s or "").split())).strip(" .,;:")


def is_attested(p, document, lang=EN):
    """Whether the document states a date inside the interval of the parsed expression `p`."""
    lang = get_language(lang)
    doc = document or ""
    a, b = p["start"], p["end"]
    sub_year = p.get("months") is not None
    range_claim = p.get("family") == "range"
    for sp in lang.spans(doc):
        q = lang.parse(sp)
        if not q:
            continue
        if range_claim:
            if (q["start"], q["end"]) == (a, b):
                return True
            continue
        if not (a <= q["start"] and q["end"] <= b):
            continue
        if sub_year:
            if q.get("months") is None or not q["months"] <= p["months"]:
                continue
            if p.get("day") is not None and q.get("day") != p["day"]:
                continue
        return True
    if sub_year:
        return False
    years = {int(m.group(1)) for m in lang.year_token.finditer(doc)}
    if range_claim:
        return a in years and b in years
    return a == b and a in years


def _verdict(ok):
    return "valid" if ok else "invalid"


def _sub_year(p, pa, r, lang):
    mc, ma, jc, ja = p["months"], pa["months"], p.get("day"), pa.get("day")
    if mc == ma and jc == ja:
        return "valid", "reformulation", "same date (sub-year)"
    if mc.isdisjoint(ma):
        return "invalid", "wrong_date", "sub-year shift (month/season)"
    if jc is not None and ja is not None:
        return "invalid", "wrong_date", "day shift"
    if jc is None and mc >= ma:
        return "valid", "generalization", "sub-year widening"
    if mc <= ma:
        ok = is_attested(p, r["document"], lang)
        return _verdict(ok), "precision", ("sub-year narrowing attested" if ok
                                           else "sub-year narrowing not attested by the document")
    return "invalid", "wrong_date", "sub-year overlap"


def judge(r, lang=EN):
    """Apply the criterion to a rewrite. Returns (verdict, class, reason); the verdict is None when the rewrite cannot be judged."""
    lang = get_language(lang)
    french = lang.name == "fr"
    if french:
        r = {**r, **{k: unicodedata.normalize("NFC", r[k]) for k in
                     ("anchor_expression", "original_sentence", "claim", "document")
                     if isinstance(r.get(k), str)}}
    ae = r.get("anchor_expression") or ""
    if lang.open_anchor.search(ae):
        return None, None, "open-ended anchor"
    txt = claim_expression(r["original_sentence"], r["claim"], lang)
    if not txt:
        return None, None, "no substitution detected"
    p = lang.parse(txt)
    if p is None:
        if MALFORMED_DECADE.search(lang.to_en(txt)):
            return None, "malformed_expression", f"invalid decade (year does not end in 0): {txt[:40]}"
        return None, None, f"unparsable expression: {txt[:40]}"
    anc = tuple(r["interval_anchor"])
    cl = (p["start"], p["end"])
    pa = lang.parse(ae) if ae else None
    if pa and (french or HUNDREDS.search(lang.to_en(ae)) or pa["family"] == "positioned"):
        anc = (pa["start"], pa["end"])
    anchor_tier = _GRANULARITY_TIER.get(r["granularity"])
    if french and pa:
        anchor_tier = pa["tier"]
    if anchor_tier is None:
        return None, None, "unknown anchor granularity"
    if anc[1] > anc[0]:
        anchor_tier = tier_from_width(anc[1] - anc[0] + 1)
    rel = "equal" if ae and _norm_expr(txt, lang) == _norm_expr(ae, lang) else relation(anc, cl)
    gap = TIER_INDEX[p["tier"]] - TIER_INDEX[anchor_tier]

    if rel == "equal":
        i_c = lang.low(r["claim"]).find(lang.low(txt))
        i_p = lang.low(r["original_sentence"]).find(lang.low(txt))
        if (i_c > 0 and lang.time_of_day.search(r["claim"][:i_c])
                and not (i_p > 0 and lang.time_of_day.search(r["original_sentence"][:i_p]))):
            m_tod = lang.time_of_day.search(r["claim"][:i_c])
            ok = lang.low(m_tod.group(0) + txt) in lang.low(r["document"])
            return _verdict(ok), "precision", ("time of day attested" if ok
                                               else "time of day added, not attested by the document")
        if p.get("months") is not None and pa is not None and pa.get("months") is not None:
            return _sub_year(p, pa, r, lang)
        if gap < 0 and p["tier"] in ("day", "month"):
            ok = is_attested(p, r["document"], lang)
            return _verdict(ok), "precision", (
                "narrowed value attested in the document (same year as the anchor)" if ok
                else "narrowing not attested by the document (same year as the anchor)")
        if gap > 0:
            return "valid", "generalization", f"widening by {gap} tier(s)"
        return "valid", "reformulation", "identical interval"
    if rel in ("disjoint", "overlaps"):
        return "invalid", "wrong_date", ("interval disjoint from the anchor" if rel == "disjoint"
                                         else "interval overlaps the anchor")
    if rel == "included":
        ok = is_attested(p, r["document"], lang)
        return _verdict(ok), "precision", ("narrowed value attested in the document" if ok
                                           else "narrowing not attested by the document")
    if not tiers.CALIBRATED:
        if ERA_CONTAINER.search(lang.to_en(txt)) or p["family"] == "era_container":
            if anchor_tier == "century":
                return "tolerated", "generalization", "era container one tier above a century anchor"
            return "invalid", "generalization", "era container without interpretive value"
        if p["family"] == "positioned":
            return "valid", "generalization", "positioned label"
        if p["family"] == "periodization":
            return "tolerated", "generalization", "unpositioned named periodization"
    elif ERA_CONTAINER.search(lang.to_en(txt)) or p["family"] == "era_container":
        return "invalid", "generalization", "era container without interpretive value"
    if p["family"] == "positioned":
        if gap <= 3:
            return "valid", "generalization", f"positioned label ({max(gap, 1)} tier(s))"
        return "invalid", "generalization", f"positioned label too wide ({gap} tiers)"
    if p["family"] == "periodization":
        if gap > 3:
            return "invalid", "generalization", f"named periodization too wide ({gap} tiers)"
        return "tolerated", "generalization", "unpositioned named periodization"
    if gap <= 3:
        return "valid", "generalization", f"widening by {max(gap, 1)} tier(s)"
    return "invalid", "generalization", f"widening by {gap} tiers"
