"""Prompts that ask an LLM to rewrite the date of a sentence."""
from __future__ import annotations

PROMPT_VERSION = "v8.7-neutral"

_LANG_NAME = {"fr": "French", "en": "English"}

_TEMPLATE = """You are an assistant specialized in analyzing temporal expressions in {lang} text.

TASK: starting from the ORIGINAL SENTENCE (in {lang}), produce a short LADDER of
rewrites that modify ONLY its temporal expression "{expr}" according to the rule
below. The rewritten sentences must stay in **{lang}**.

ABSOLUTE RULE: for each rewrite, copy the original sentence WORD FOR WORD and
replace only "{expr}". No other word may change, disappear, or be added. No new
fact, no new entity, no rephrasing beyond the temporal expression itself. Each
result must be a grammatically correct, natural {lang} sentence. If you cannot
make the substitution cleanly, produce one fewer rewrite instead.

SOURCE TEXT (ground truth: faithfulness is judged against it{pub_clause}):
\"\"\"
{source}
\"\"\"

ORIGINAL SENTENCE: "{sentence}"
Temporal expression to replace: "{expr}"

{instructions}

{count_clause}

REPLY ONLY with a JSON array of strings, one full rewritten sentence per string,
ORDERED as the ladder progresses. No keys, no objects, no Markdown, no commentary.
Expected shape: ["rewritten sentence 1", "rewritten sentence 2", "rewritten sentence 3"]
"""

_EX_HEAD = (
    "WORKED EXAMPLES for this series (replacement expressions only; your output must be"
    " the full original sentence with only the temporal expression replaced):\n"
)

_EXAMPLES: dict[str, dict[tuple[str, str], str]] = {
    "fr": {
        ("generalization", "valid"): """\
  source "en 1294", expression "1294" -> ["la décennie 1290", "la fin du XIIIe siècle", "le XIIIe siècle"]
  source "le 8 juin 1984", expression "le 8 juin 1984" -> ["juin 1984", "1984", "les années 1980"]
  source "en 1685" (règne de Louis XIV), expression "1685" -> ["les années 1680", "le règne de Louis XIV"]""",
        ("generalization", "invalid"): """\
  source "en 1294", expression "1294" -> ["les années 1280", "le XIVe siècle", "le Moyen Âge"]
  source "le 8 juin 1984", expression "le 8 juin 1984" -> ["les années 1970", "le XXe siècle", "l'époque contemporaine"]
  source "en 1685" (règne de Louis XIV), expression "1685" -> ["sous le règne de Louis XV"]""",
        ("precision", "valid"): """\
  source "en 1294", expression "la fin du XIIIe siècle" -> ["la décennie 1290", "1294"]
  source "le 8 juin 1984", expression "les années 1980" -> ["1984", "juin 1984", "le 8 juin 1984"]
  source only says "dans les années 1980", expression "le XXe siècle" -> ["les années 1980"]""",
        ("precision", "invalid"): """\
  source "en 1294", expression "la fin du XIIIe siècle" -> ["1296", "le printemps 1294", "les années 1310"]
  source "le 8 juin 1984", expression "les années 1980" -> ["1987", "le 8 juin 1986", "le début des années 1990"]""",
        ("reformulation", "valid"): """\
  expression "le XIIIe siècle" -> ["le treizième siècle", "le siècle allant de 1201 à 1300"]
  expression "les années 1980" -> ["la décennie 1980", "entre 1980 et 1989"]""",
        ("reformulation", "invalid"): """\
  expression "le XIIIe siècle" -> ["vers 1250", "le XIVe siècle", "la fin du Moyen Âge"]
  expression "les années 1980" -> ["de 1985 à 1995", "les années 1970"]""",
        ("wrong_date", "invalid"): """\
  source "en 1294", expression "1294" -> ["1295", "1284", "les années 1280", "le 8 août 1294"]
    (subtle shifts first, then an invented precise value)
  source "le 8 juin 1984", expression "le 8 juin 1984" -> ["le 7 juin 1984", "le 8 août 1984", "le 8 juin 1985"]
  source "en 1685" (règne de Louis XIV), expression "1685" -> ["sous Louis XIII", "sous la Régence"]""",
    },
    "en": {
        ("generalization", "valid"): """\
  source "in 1294", expression "1294" -> ["the 1290s", "the late 13th century", "the 13th century"]
  source "on 8 June 1984", expression "8 June 1984" -> ["June 1984", "1984", "the 1980s"]
  source "in 1685" (reign of Louis XIV), expression "1685" -> ["the 1680s", "the reign of Louis XIV"]""",
        ("generalization", "invalid"): """\
  source "in 1294", expression "1294" -> ["the 1280s", "the 14th century", "the Middle Ages"]
  source "on 8 June 1984", expression "8 June 1984" -> ["the 1970s", "the 20th century", "the modern era"]
  source "in 1685" (reign of Louis XIV), expression "1685" -> ["the reign of Louis XV"]""",
        ("precision", "valid"): """\
  source "in 1294", expression "the late 13th century" -> ["the 1290s", "1294"]
  source "on 8 June 1984", expression "the 1980s" -> ["1984", "June 1984", "8 June 1984"]
  source only says "in the 1980s", expression "the 20th century" -> ["the 1980s"]""",
        ("precision", "invalid"): """\
  source "in 1294", expression "the late 13th century" -> ["1296", "the spring of 1294", "the 1310s"]
  source "on 8 June 1984", expression "the 1980s" -> ["1987", "8 June 1986", "the early 1990s"]""",
        ("reformulation", "valid"): """\
  expression "the 13th century" -> ["the thirteenth century", "the 1200s", "the century from 1201 to 1300"]
  expression "the 1980s" -> ["the decade of the 1980s", "between 1980 and 1989"]""",
        ("reformulation", "invalid"): """\
  expression "the 13th century" -> ["around 1250", "the 14th century", "the late Middle Ages"]
  expression "the 1980s" -> ["from 1985 to 1995", "the 1970s"]""",
        ("wrong_date", "invalid"): """\
  source "in 1294", expression "1294" -> ["1295", "1284", "the 1280s", "8 August 1294"]
    (subtle shifts first, then an invented precise value)
  source "on 8 June 1984", expression "8 June 1984" -> ["7 June 1984", "8 August 1984", "8 June 1985"]
  source "in 1685" (reign of Louis XIV), expression "1685" -> ["under Louis XIII", "during the Regency"]""",
    },
}


def examples_for(error_type: str, polarity: str, lang: str = "fr") -> str:
    return _EX_HEAD + _EXAMPLES.get(lang, _EXAMPLES["fr"])[(error_type, polarity)]


_FORMS = {
    "fr": ('une décennie, un siècle, une moitié ou un quart de siècle, un intervalle explicite'
           ' (« entre X et Y », « de X à Y »), une saison ou un mois, une expression'
           ' positionnée (« le début / le milieu / la fin de ... », « le tournant du ... »),'
           ' une approximation (« vers », « aux alentours de »), une période nommée'),
    "en": ('a decade, a century, a half or quarter of a century, an explicit range'
           ' ("between X and Y", "from X to Y"), a season or month, a positioned expression'
           ' ("the early / mid / late ...", "the turn of the ..."), an approximation'
           ' ("around", "circa"), a named period'),
}

_BORDERLINE = {
    "valid": ("BORDERLINE CASES: include some rewrites that are only just faithful — as far"
              " as you can go while still being faithful."),
    "invalid": ("BORDERLINE CASES: include some rewrites that are only just wrong — very close"
                " to a faithful rewrite, easy to mistake for a correct one."),
}


def variety_for(polarity: str, lang: str = "fr") -> str:
    return ("VARIETY: across the rewrites, vary the FORM of the new temporal expression"
            f" where it fits the task — e.g. {_FORMS.get(lang, _FORMS['fr'])}. Do not reuse"
            " the wordings of the examples.\n" + _BORDERLINE[polarity])


CLASSES: dict[str, dict] = {
    "generalization": {
        "can_be_valid": True,
        "instr": (
            'GENERALIZATION LADDER: replace "{expr}" with a sequence of WIDER periods,'
            " each one containing the previous. See the examples above.\n"
            "- FAITHFUL series: every rung must contain the original period and be wider"
            " than the previous rung (no repeats, no going back down).\n"
            "- NOT FAITHFUL series: produce rungs that look like a generalization of"
            ' "{expr}" but are not faithful to it.\n'
            "FORBIDDEN in the faithful series: narrowing the period or shifting it."
        ),
    },
    "precision": {
        "can_be_valid": True,
        "instr": (
            'PRECISION LADDER: replace "{expr}" with a sequence of MORE PRECISE dates,'
            " each one narrower than the previous and included within the original"
            " interval. See the examples above.\n"
            "- FAITHFUL series: every rung must be EXPLICITLY supported by the SOURCE"
            " TEXT above — stop the ladder as soon as you would need to invent a level"
            " of detail the source does not state. A short, honest ladder (even a single"
            " rung) beats a long invented one.\n"
            "- NOT FAITHFUL series: precise dates absent from the source (invented), or"
            " outside the original interval."
        ),
    },
    "reformulation": {
        "can_be_valid": True,
        "instr": (
            'REFORMULATION: express EXACTLY the same interval as "{expr}" in different'
            " words, several distinct natural phrasings. See the examples above.\n"
            "- FAITHFUL: identical interval, different wording.\n"
            "- NOT FAITHFUL: the rephrasing changes the interval, even slightly."
        ),
    },
    "wrong_date": {
        "can_be_valid": False,
        "instr": (
            'WRONG DATE: rewrite the sentence with a FALSE date, NOT supported by the'
            ' source text. Produce a MIX of the two kinds shown in the examples above'
            " (roughly half and half, in this order):\n"
            "  1) SUBTLE, PLAUSIBLE shifts at the same granularity as \"{expr}\" — off by"
            " a day, a month, a year, or one tier (these are the most useful: close"
            " enough to look right, but factually wrong).\n"
            "  2) INVENTED PRECISE values — add a level of detail (day, month, exact"
            " date) that the source never states, even if the broad period happens to"
            " match.\n"
            "Every rewrite must be erroneous."
        ),
    },
}


def polarities_for(error_type: str) -> list[str]:
    return ["valid", "invalid"] if CLASSES[error_type]["can_be_valid"] else ["invalid"]


def label_llm_for(polarity: str) -> bool:
    return polarity == "valid"


def build(
    error_type: str,
    polarity: str,
    *,
    anchor_expr: str,
    sentence: str,
    source: str,
    n: int,
    source_pub_year: int | None = None,
    lang: str = "fr",
) -> str:
    """The prompt asking for rewrites of one type and validity of an anchor's date."""
    spec = CLASSES[error_type]
    instr = examples_for(error_type, polarity, lang) + "\n\n" + spec["instr"].format(expr=anchor_expr)
    instr += "\n\n" + variety_for(polarity, lang)

    if not spec["can_be_valid"]:
        count_clause = f"Produce up to {n} rewrites (fewer is fine if the mix above is honestly covered), ALL of them erroneous."
    elif polarity == "valid":
        instr += "\nIn this series, produce ONLY FAITHFUL rewrites, ordered as a ladder."
        count_clause = f"Produce up to {n} FAITHFUL rewrites, one per natural rung of the ladder (a short honest ladder beats a padded one)."
    else:
        instr += "\nIn this series, produce ONLY NOT FAITHFUL rewrites."
        count_clause = f"Produce up to {n} NOT FAITHFUL rewrites."

    pub_clause = (
        f"; published in {source_pub_year}, it cannot assert anything later"
        if source_pub_year
        else ""
    )
    return _TEMPLATE.format(
        lang=_LANG_NAME.get(lang, "French"),
        expr=anchor_expr,
        sentence=sentence.strip(),
        source=source.strip()[:4000],
        instructions=instr,
        count_clause=count_clause,
        pub_clause=pub_clause,
    )
