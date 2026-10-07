"""French layer of the tier parser: find the expressions and translate them into English ones."""
import re
import unicodedata

from . import tiers


def fold(s):
    out = []
    for c in s or "":
        if c in "’‘ʼ`´":
            out.append("'")
            continue
        d = unicodedata.normalize("NFKD", c)
        b = d[0] if d and d[0].isascii() else c
        lb = b.lower()
        out.append(lb if len(lb) == 1 else b)
    return "".join(out)


class FoldedRegex:

    def __init__(self, pattern, flags=0):
        self.rx = re.compile(pattern, flags)

    def search(self, s):
        return self.rx.search(fold(s))

    def finditer(self, s):
        return self.rx.finditer(fold(s))


MONTHS = {"janvier": "january", "fevrier": "february", "mars": "march", "avril": "april",
          "mai": "may", "juin": "june", "juillet": "july", "aout": "august",
          "septembre": "september", "octobre": "october", "novembre": "november",
          "decembre": "december"}
SEASONS = {"printemps": "spring", "ete": "summer", "automne": "autumn", "hiver": "winter"}
_MONTH = "(?:" + "|".join(MONTHS) + ")"
_SEASON = "(?:" + "|".join(SEASONS) + ")"

ORD_FR = {"premier": 1, "1er": 1, "deuxieme": 2, "second": 2, "troisieme": 3,
          "quatrieme": 4, "cinquieme": 5, "sixieme": 6, "septieme": 7, "huitieme": 8,
          "neuvieme": 9, "dixieme": 10, "onzieme": 11, "douzieme": 12, "treizieme": 13,
          "quatorzieme": 14, "quinzieme": 15, "seizieme": 16, "dix-septieme": 17,
          "dix-huitieme": 18, "dix-neuvieme": 19, "vingtieme": 20,
          "vingt et unieme": 21, "vingt-et-unieme": 21, "vingt-unieme": 21}
_ORDW = "(?:" + "|".join(sorted((re.escape(k) for k in ORD_FR), key=len, reverse=True)) + ")"
_ROMO = r"(?:[ivx]{1,5}(?:\s?(?:ere|er|re|eme|ieme|e))?)"
_CORD = rf"(?:{_ORDW}|{_ROMO})"

ERAS_FR = {
    "haut moyen age": "the early middle ages", "haut-moyen age": "the early middle ages",
    "moyen age central": "the high middle ages", "moyen age classique": "the high middle ages",
    "plein moyen age": "the high middle ages",
    "bas moyen age": "the late middle ages", "moyen age tardif": "the late middle ages",
    "moyen age": "the middle ages", "moyen-age": "the middle ages",
    "epoque medievale": "the medieval period", "periode medievale": "the medieval period",
    "antiquite tardive": "late antiquity", "antiquite classique": "classical antiquity",
    "antiquite": "antiquity",
    "epoques moderne et contemporaine": "the early modern period and the contemporary period",
    "periodes moderne et contemporaine": "the early modern period and the contemporary period",
    "epoque moderne": "the early modern period", "periode moderne": "the early modern period",
    "temps modernes": "the early modern period",
    "epoque contemporaine": "the contemporary period",
    "periode contemporaine": "the contemporary period",
    "ancien regime": "the ancien regime",
    "renaissance": "the renaissance",
    "siecle des lumieres": "the enlightenment", "epoque des lumieres": "the enlightenment",
    "ere industrielle": "the industrial era", "epoque industrielle": "the industrial era",
    "epoque carolingienne": "the carolingian era",
    "guerre froide": "the cold war", "belle epoque": "the belle epoque",
    "entre-deux-guerres": "the interwar period",
    "epoque romantique": "the romantic period", "epoque baroque": "the baroque period",
    "revolution scientifique": "the scientific revolution",
    "epoque victorienne": "the victorian era",
    "empire romain": "the roman empire", "prehistoire": "prehistory",
}
_ERA_NAMES = sorted(ERAS_FR, key=len, reverse=True)

_POS_WORDS = (
    r"(?:(?:tout |toute )?(?:debut|fin|milieu|mitan|coeur|aube|oree|crepuscule|tournant|"
    r"charniere)|(?:premiere|seconde|deuxieme|derniere) moitie|"
    r"(?:premier|second|deuxieme|troisieme|dernier) tiers|"
    r"(?:premier|second|deuxieme|troisieme|quatrieme|dernier) quart|"
    r"(?:(?:vingt-cinq|25)\s+)?(?:premieres|dernieres) (?:annees|decennies)|"
    r"(?:premiere|derniere) decennie)")
_POS = rf"(?:{_POS_WORDS}\s+(?:du|de la|de l'|des|d')\s*)?"
_BC = r"(?:av\.?|avant)\s*(?:j\.?\s*-?\s*c\.?|jesus[- ]christ|notre ere|n\.\s*e\.?)"
_AD = r"(?:(?:ap\.?|apres)\s*(?:j\.?\s*-?\s*c\.?|jesus[- ]christ)|de notre ere)"
_SFX = rf"(?:\s*(?:{_BC}|{_AD}))?"
_LIST = r"(?:\s*[,\-–—]\s*|\s+(?:et|a|au)\s+)"

_CENT = (rf"{_POS}(?:le |les |l')?{_CORD}(?:{_LIST}(?:le |l')?{_CORD})*\s+siecles?{_SFX}")
_YEAR_AN = r"(?:l'an(?:nee)?\s+)?\d{1,4}"
SPAN = re.compile(
    r"(?<!\w)("
    + rf"(?:entre|de|du|depuis)\s+(?:la |le |l')?{_CENT}\s+(?:et|a|au|jusqu'au|jusqu'a)\s+"
      rf"(?:la |le |l')?{_CENT}"
    + rf"|(?:entre|du|de)\s+(?:le |la |l')?{_POS}{_CORD}\s+(?:et|au|a)\s+(?:le |la |l')?"
      rf"{_CENT}"
    + rf"|{_CENT}"
    + r"|(?:(?:debut|fin|milieu) des )?annees\s+\d{2,4}(?:" + _LIST
    + r"(?:les |des )?(?:annees\s+)?\d{2,4})?" + _SFX
    + r"|(?:(?:premiere|derniere) )?decennie\s+\d{3,4}"
    + rf"|(?:1er|\d{{1,2}})\s+(?:du mois de\s+|de\s+)?{_MONTH}\s+(?:de l'(?:annee|an)\s+)?\d{{1,4}}{_SFX}"
    + rf"|(?:mois d[e']\s*)?(?:{_MONTH}|{_SEASON})\s+(?:de l'(?:annee|an)\s+|de\s+|d')?\d{{1,4}}{_SFX}"
    + rf"|entre\s+{_YEAR_AN}\s+et\s+{_YEAR_AN}{_SFX}"
    + r"|(?:de\s+)?\d{3,4}(?:\s*[-–—]\s*|\s+a\s+)\d{2,4}(?![-\d])" + _SFX
    + rf"|{_POS}(?:le |l')?(?:premier|deuxieme|second|troisieme)\s+millenaire{_SFX}"
    + rf"|{_POS}(?:le |la |l')?(?:" + "|".join(re.escape(n) for n in _ERA_NAMES) + r")(?![\w-])"
    + rf"|l'an(?:nee)?\s+(?:mil(?:le)?|\d{{1,4}}){_SFX}"
    + rf"|\d{{1,4}}\s*(?:{_BC}|{_AD})"
    + r"|\d{3,4}(?![\d])"
    + r")(?![\w])")


def spans(text):
    f = fold(text)
    out = []
    for m in SPAN.finditer(f):
        s = (text or "")[m.start(1):m.end(1)].strip(" ,.;:-")
        if s:
            out.append(s)
    return out


_POS_EN = [
    (r"(?:la )?premiere moitie", "first half of the"),
    (r"(?:la )?(?:seconde|deuxieme|derniere) moitie", "second half of the"),
    (r"(?:le )?premier tiers", "first third of the"),
    (r"(?:le )?(?:second|deuxieme) tiers", "second third of the"),
    (r"(?:le )?(?:dernier|troisieme) tiers", "last third of the"),
    (r"(?:le )?premier quart", "first quarter of the"),
    (r"(?:le )?(?:second|deuxieme) quart", "second quarter of the"),
    (r"(?:le )?troisieme quart", "third quarter of the"),
    (r"(?:le )?(?:dernier|quatrieme) quart", "last quarter of the"),
    (r"(?:les )?(?:vingt-cinq|25) premieres annees", "first quarter of the"),
    (r"(?:les )?(?:vingt-cinq|25) dernieres annees", "last quarter of the"),
    (r"(?:les )?premieres annees", "first years of the"),
    (r"(?:les )?dernieres annees", "last years of the"),
    (r"(?:les )?premieres decennies", "first decades of the"),
    (r"(?:les )?dernieres decennies", "last decades of the"),
    (r"(?:la )?premiere decennie", "first decade of the"),
    (r"(?:la )?derniere decennie", "final decade of the"),
    (r"(?:le )?tout debut", "very beginning of the"),
    (r"(?:la )?toute fin", "very end of the"),
    (r"(?:le )?debut", "beginning of the"),
    (r"(?:la )?fin", "end of the"),
    (r"(?:le )?(?:milieu|mitan|coeur)", "middle of the"),
    (r"(?:l')?(?:aube|oree)", "dawn of the"),
    (r"(?:le )?crepuscule", "twilight of the"),
    (r"(?:le |la )?(?:tournant|charniere)", "turn of the"),
]
_LINK = r"\s+(?:du|de la|de l'|des|d')\s*"


def _rom(s):
    v = {"i": 1, "v": 5, "x": 10}
    n = 0
    for a, b in zip(s, s[1:] + " "):
        n += -v[a] if v.get(b, 0) > v[a] else v[a]
    return n


def to_en(expr):
    """Translate a French temporal expression into its English equivalent so that the English parser can read it."""
    if expr is None:
        return None
    t = " " + " ".join(fold(unicodedata.normalize("NFC", str(expr))).split()) + " "
    t = t.replace("ᵉ", "e")
    t = re.sub(r"\s*\b" + _BC, " bc ", t)
    t = re.sub(r"\s*\b" + _AD, " ad ", t)
    t = re.sub(r"\bl'an\s+mil(?:le)?\b", " 1000", t)
    t = re.sub(r"\b(?:de\s+)?l'(?:an|annee)\s+(?=\d)", " ", t)
    t = re.sub(r"\b(?:du\s+)?mois\s+d(?:e\s+|')", " ", t)
    t = re.sub(r"\b1er\b", "1", t)
    for fr, en in {**MONTHS, **SEASONS}.items():
        t = re.sub(rf"\b{fr}\b", en, t)
    t = re.sub(r"\b(spring|summer|autumn|winter|" + "|".join(MONTHS.values())
               + r")\s+(?:de\s+|d')(?=\d)", r"\1 ", t)
    t = re.sub(r"\bl'(summer|autumn|winter)\b", r"\1", t)
    t = re.sub(r"\b(?:les\s+|des\s+)?annees\s+(\d{3,4})\s*(?:a|[-–—])\s*(\d{3,4})\b",
               r" \1 to \2 ", t)
    t = re.sub(r"\b(?:le\s+|au\s+|du\s+)?debut\s+des\s+(?=annees)", " early ", t)
    t = re.sub(r"\b(?:la\s+|a la\s+|de la\s+)?fin\s+des\s+(?=annees)", " late ", t)
    t = re.sub(r"\b(?:le\s+|au\s+|du\s+)?milieu\s+des\s+(?=annees)", " mid ", t)

    def _dec(m):
        n = m.group(1)
        era = re.match(r"\s*(?:ad|bc)\b", t[m.end():])
        if len(n) == 2 and not era:
            n = "19" + n
        return f" {n}s"
    t = re.sub(r"\b(?:les\s+|des\s+|aux\s+)?annees\s+(\d{2,4})\b", _dec, t)
    t = re.sub(r"\b(?:au cours de\s+)?(?:la\s+)?decennie\s+(\d{3,4})\b", r" \1s", t)
    for fr, en in _POS_EN:
        t = re.sub(r"\b" + fr + _LINK, f" {en} ", t)
    for fr in _ERA_NAMES:
        t = re.sub(rf"(?:\b(?:le|la|l')\s*)?\b{re.escape(fr)}(?![\w-])", f" {ERAS_FR[fr]} ", t)
    if re.search(r"siecle|millenaire", t):
        for w, n in sorted(ORD_FR.items(), key=lambda kv: -len(kv[0])):
            t = re.sub(rf"(?<![\w-]){re.escape(w)}\b(?!\s+(?:half|third|quarter|years|decade))",
                       f"{n}th", t)
        t = re.sub(r"\b([ivx]{1,5})\s?(?:ere|er|re|eme|ieme|e)\b",
                   lambda m: f"{_rom(m.group(1))}th", t)
        t = re.sub(r"\b([ivx]{1,5})\b(?=" + _LIST + r"(?:le |l')?\d+th)",
                   lambda m: f"{_rom(m.group(1))}th", t)
        t = re.sub(r"\b([123])th\s+millenaires?\b",
                   lambda m: {"1": " the first", "2": " the second", "3": " the third"}[m.group(1)]
                   + " millennium", t)
        t = re.sub(r"\bsiecles\b", "centuries", t)
        t = re.sub(r"\bsiecle\b", "century", t)
    t = re.sub(r"\b(?:vers|autour de|aux alentours de|aux environs de|pres de|environ|"
               r"circa|ca\.|c\.)\s*", " around ", t)
    t = re.sub(r"^\s*(?:(?:au cours de|au cours du|durant|pendant|dans|en|au|aux|a|le|la|"
               r"les|l'|sous|lors de|around)\s+)+", " ", t)
    t = re.sub(r"\bentre\s+(?:le\s+|la\s+|l')?", " between ", t)
    if re.search(r"\d", t):
        if re.search(r"\s(?:a|au|aux|jusqu'au|jusqu'a|jusqu'en)\s", t):
            t = re.sub(r"^\s*(?:periode\s+)?(?:allant\s+)?(?:de|du|des|depuis)\s+"
                       r"(?:la\s+|le\s+|l')?", " from ", t)
            t = re.sub(r"\s(?:a|au|aux|jusqu'au|jusqu'a|jusqu'en)\s+(?:la\s+|le\s+|l')?", " to ", t)
        t = re.sub(r"\s+et\s+(?:le\s+|la\s+|l')?", " and ", t)
    t = re.sub(r"^\s*(?:(?:du|de|des|d'|periode)\s+)+", " ", t)
    t = re.sub(r"\bthe\s+the\b", "the", t)
    t = " ".join(t.split())
    return t or None


def parse(expr):
    return tiers.parse(to_en(expr))


OPEN_ANCHOR = FoldedRegex(r"^\s*(?:a partir d[eu']|jusqu'(?:a|au|en)\b|depuis\b|avant\b|apres\b|"
                          r"des\s+(?=\d)|anterieure?s?\s+a\b|posterieure?s?\s+a\b|au-dela\b)")
TIME_OF_DAY = FoldedRegex(r"\b(matin|matinee|apres-midi|soir|soiree|nuit|midi|minuit)\s+"
                          r"(?:du|de|d')\s*(?:le\s+)?$")
TEMPORAL_HINT = FoldedRegex(r"\d{3,4}|siecle|decennie|regne|dynastie|\bere\b|epoque|\bage\b|"
                            r"periode|millenaire")
YEAR_TOKEN = FoldedRegex(r"(?<![,.\d$£€])\b(\d{3,4})\b(?!\s*%|[.,]\d|\s*" + _BC + r"|\s+\d{3}\b)")
