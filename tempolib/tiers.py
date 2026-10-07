"""Parse English temporal expressions into an interval, a granularity tier and a family."""
import os
import re

CALIBRATED = os.environ.get("NO_CALIBRATION") != "1"

TIERS = ["day", "month", "year", "decade", "half_century", "century", "period"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
       "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
       "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
       "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twentyfirst": 21}
ERAS = {
    "antiquity": (-800, 500, "era_container"), "classical antiquity": (-800, 500, "era_container"),
    "late antiquity": (200, 600, "periodization"),
    "the middle ages": (500, 1500, "era_container"), "middle ages": (500, 1500, "era_container"),
    "the medieval period": (500, 1500, "era_container"), "medieval period": (500, 1500, "era_container"),
    "the early middle ages": (500, 1000, "periodization"), "the high middle ages": (1000, 1250, "periodization"),
    "the late middle ages": (1250, 1500, "periodization"),
    "the renaissance": (1400, 1600, "periodization"), "renaissance": (1400, 1600, "periodization"),
    "the early modern period": (1500, 1800, "era_container"), "early modern period": (1500, 1800, "era_container"),
    "the modern period": (1500, 2000, "era_container"), "the modern era": (1500, 2000, "era_container"),
    "modern era": (1500, 2000, "era_container"), "the early modern era": (1500, 1800, "era_container"),
    "early modern era": (1500, 1800, "era_container"), "the victorian era": (1837, 1901, "periodization"),
    "the belle epoque": (1871, 1914, "periodization"), "the interwar period": (1918, 1939, "periodization"),
    "the contemporary period": (1800, 2100, "era_container"),
    "the enlightenment": (1650, 1800, "periodization"),
    "the industrial era": (1760, 1900, "periodization"),
    "the carolingian era": (751, 987, "periodization"), "the reformation": (1517, 1648, "periodization"),
    "the age of revolutions": (1774, 1849, "periodization"), "the cold war": (1947, 1991, "periodization"),
    "the gilded age": (1870, 1900, "periodization"), "the romantic period": (1800, 1850, "periodization"),
    "the baroque period": (1600, 1750, "periodization"), "the scientific revolution": (1543, 1687, "periodization"),
    "the second millennium": (1000, 2000, "era_container"), "the first millennium": (0, 1000, "era_container"),
    "the medieval era": (500, 1500, "era_container"), "medieval era": (500, 1500, "era_container"),
    "the georgian era": (1714, 1837, "periodization"),
    "the regency era": (1811, 1820, "periodization"), "the regency period": (1811, 1820, "periodization"),
    "the edwardian era": (1901, 1910, "periodization"),
    "the progressive era": (1896, 1917, "periodization"),
    "the romantic era": (1800, 1850, "periodization"),
    "the ancien regime": (1589, 1789, "era_container"), "the ancien régime": (1589, 1789, "era_container"),
}
for _lab in list(ERAS):
    if _lab.startswith("the "):
        _bare = _lab[4:]
        ERAS.setdefault(_bare, ERAS[_lab])
del _lab
POSITIONED = re.compile(r"\b(early|late|mid|middle|beginning|end|turn|first half|second half|"
                        r"first third|second third|last third|first quarter|second quarter|"
                        r"third quarter|last quarter|opening|closing|final|first years|last years|"
                        r"earliest years|final years|first decades|opening decades|early decades|"
                        r"earliest decades|last decades|closing decades|final decades|"
                        r"start|dawn|twilight)\b", re.I)

_MONTHS = (r"january|february|march|april|may|june|july|august|september|october|"
           r"november|december")

_MONTH_NUM = {m: i for i, m in enumerate(_MONTHS.split("|"), 1)}
_SEASONS = {"spring": frozenset({3, 4, 5}), "summer": frozenset({6, 7, 8}),
            "autumn": frozenset({9, 10, 11}), "fall": frozenset({9, 10, 11}),
            "winter": frozenset({12, 1, 2})}

_CONNECT = re.compile(r"\s+(?:to|and|until|till|through)\s+|\s*[–—]\s*|(?<=\d)\s*-\s*(?=\d)|"
                      r"(?<=\dst|\dnd|\drd|\dth|\d0s)\s*-\s*(?=\S)")


def _num_century(t):
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", t)
    if m:
        return int(m.group(1))
    for w, v in sorted(ORD.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"(?<![\w-])" + re.escape(w) + r"\b", t):
            return v
    return None


def tier_from_width(w):
    if w <= 1:
        return "year"
    if w <= 12:
        return "decade"
    if w <= 60:
        return "half_century"
    if w <= 130:
        return "century"
    return "period"


def _range(t):
    """Interval covered by a range such as 1840-1914 or the 12th to the 14th century."""
    m = _CONNECT.search(t)
    if not m:
        return None
    left = re.sub(r"^(?:from|between)\s+", "", t[:m.start()].strip())
    right = t[m.end():].strip()
    if not left or not right:
        return None
    if "centur" in right and "centur" not in left:
        left += " century"
    if re.search(r"\b(bc|bce)\b", right) and not re.search(r"\b(bc|bce)\b", left):
        left += " bc"
    if re.fullmatch(r"\d{3,4}", left) and re.fullmatch(r"\d{2}", right):
        right = left[:-2] + right
    pl, pr = parse(left), parse(right)
    if not pl or not pr:
        return None
    if re.search(r"\b(turn|junction|cusp)\s+of\b", left) and "centur" in right:
        n = _num_century(left)
        if n:
            return dict(start=n * 100 - 10, end=n * 100 + 10, tier="decade", family="positioned")
    if re.fullmatch(r"\s+and\s+", m.group(0)) and \
            max(pl["start"], pr["start"]) - min(pl["end"], pr["end"]) > 1:
        return None
    a, b = min(pl["start"], pr["start"]), max(pl["end"], pr["end"])
    return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="range")


def parse(expr):
    """Parse an English temporal expression into its interval, tier and family; None if it cannot be parsed."""
    if expr is None:
        return None
    t = " ".join(str(expr).lower().split())
    t = t.replace("’", "'").strip(" .,;:")
    t = re.sub(r"\b(mid|early|late)-(?=[a-z])", r"\1 ", t)
    bc = bool(re.search(r"\b(bc|bce|b\.c\.|before christ|before our era)\b", t))

    m = re.search(r"\b(\d{3,4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b", t)
    if m:
        y = int(m.group(1))
        if bc:
            y = -y
        return dict(start=y, end=y, tier="day", family="simple",
                    months=frozenset({int(m.group(2))}), day=int(m.group(3)))

    rg = _range(t)
    if rg:
        return rg

    m = re.search(r"\b(\d{1,2})00s\b", t)
    if m and int(m.group(1)) < 20 and not re.search(r"\bdecade\b", t):
        return parse(t[:m.start()] + f"{int(m.group(1)) + 1}th century" + t[m.end():])

    for lab, (a, b, fam) in sorted(ERAS.items(), key=lambda kv: -len(kv[0])):
        if not CALIBRATED and "ancien r" in lab:
            continue
        mm = re.search(r"\b" + re.escape(lab) + r"\b", t)
        if mm:
            rest = t[:mm.start()] + " " + t[mm.end():]
            if POSITIONED.search(rest):
                third = (b - a) // 3
                if re.search(r"\b(early|beginning|opening|first|start|dawn)\b", rest):
                    a, b = a, a + third
                elif re.search(r"\b(late|end|closing|final|last|twilight)\b", rest):
                    a, b = b - third, b
                else:
                    a, b = a + third, b - third
                return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="positioned")
            return dict(start=a, end=b, tier="period", family=fam)

    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS})\s+(\d{{3,4}})\b", t)
    if m:
        d, mo, y = int(m.group(1)), m.group(2), int(m.group(3))
    else:
        m = re.search(rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{3,4}})\b", t)
        if m:
            mo, d, y = m.group(1), int(m.group(2)), int(m.group(3))
    if m:
        if bc:
            y = -y
        return dict(start=y, end=y, tier="day", family="simple",
                    months=frozenset({_MONTH_NUM[mo]}), day=d)

    m = re.search(rf"\b({_MONTHS}|spring|summer|autumn|fall|winter)\s+(?:of\s+)?(\d{{3,4}})\b", t)
    if m:
        y = int(m.group(2))
        if bc:
            y = -y
        mo = m.group(1)
        return dict(start=y, end=y, tier="month", family="simple",
                    months=_SEASONS.get(mo) or frozenset({_MONTH_NUM[mo]}), day=None)

    m = re.search(r"\b(\d{3,4})\s*(?:-|–|to|and)\s*(\d{3,4})\b", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            a, b = b, a
        return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="range")

    m = re.search(r"\b(\d{2,4})0s\b", t)
    if m:
        a = int(m.group(1)) * 10
        if bc:
            return dict(start=-(a + 9), end=-a, tier="decade", family="simple")
        if re.search(r"\b(early)\b", t):
            return dict(start=a, end=a + 5, tier="decade", family="positioned")
        if re.search(r"\b(late)\b", t):
            return dict(start=a + 4, end=a + 9, tier="decade", family="positioned")
        if re.search(r"\b(mid)\b", t):
            return dict(start=a + 3, end=a + 7, tier="decade", family="positioned")
        return dict(start=a, end=a + 9, tier="decade", family="simple")

    m2 = re.search(r"([a-z]+(?:-[a-z]+)?)\s+(?:and|to)\s+([a-z]+(?:-[a-z]+)?)[\s-]+centur", t)
    cs = re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)\b", t)
    if len(cs) < 2 and m2 and m2.group(1) in ORD and m2.group(2) in ORD:
        v1, v2 = ORD[m2.group(1)], ORD[m2.group(2)]
        a, b = (min(v1, v2) - 1) * 100 + 1, max(v1, v2) * 100
        return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="range")
    if len(cs) >= 2 and "centur" in t:
        a, b = (int(cs[0]) - 1) * 100 + 1, int(cs[-1]) * 100
        return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="range")
    if "centur" in t:
        n = _num_century(t)
        if n:
            a, b = (n - 1) * 100 + 1, n * 100
            if bc:
                a, b = -b, -a
            if POSITIONED.search(t):
                if re.search(r"\bturn\b", t):
                    return dict(start=a - 10, end=a + 10, tier="decade", family="positioned")
                if re.search(r"\b(first|earliest|opening)\s+years\b", t):
                    b = a + 9
                elif re.search(r"\b(last|final|closing)\s+years\b", t):
                    a = b - 9
                elif re.search(r"\b(first|earliest|early|opening)\s+decades\b", t):
                    b = a + 32
                elif re.search(r"\b(last|final|closing|late)\s+decades\b", t):
                    a = a + 66
                elif re.search(r"\bfirst quarter\b", t):
                    b = a + 24
                elif re.search(r"\bsecond quarter\b", t):
                    a, b = a + 25, a + 49
                elif re.search(r"\bthird quarter\b", t):
                    a, b = a + 50, a + 74
                elif re.search(r"\blast quarter\b", t):
                    a = a + 75
                elif re.search(r"\bfirst third\b", t):
                    b = a + 32
                elif re.search(r"\bsecond third\b", t):
                    a, b = a + 33, a + 65
                elif re.search(r"\blast third\b", t):
                    a = a + 66
                elif CALIBRATED and re.search(r"\bdawn\b", t):
                    b = a + 9
                elif CALIBRATED and re.search(r"\btwilight\b", t):
                    a = b - 9
                elif re.search(r"\b(early|beginning|opening|first half|start|dawn)\b", t):
                    b = a + 49
                elif re.search(r"\b(late|end|closing|second half|final|twilight)\b", t):
                    a = b - 49
                else:
                    a, b = a + 33, b - 33
                return dict(start=a, end=b, tier=tier_from_width(b - a + 1), family="positioned")
            return dict(start=a, end=b, tier="century", family="simple")

    m = re.search(r"\b(?:ad|a\.d\.)\s+(\d{1,4})\b|\b(\d{1,4})\s*(?:bce|bc|b\.c\.|ce|ad|a\.d\.)(?!\w)", t)
    if m:
        y = int(m.group(1) or m.group(2))
        if bc:
            y = -y
        return dict(start=y, end=y, tier="year", family="simple")

    m = re.search(r"\b(\d{3,4})\b", t)
    if m:
        y = int(m.group(1))
        if bc:
            y = -y
        return dict(start=y, end=y, tier="year", family="simple")
    return None
