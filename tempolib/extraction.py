"""Pick the temporal-expression extractor of a language."""
from __future__ import annotations

from types import ModuleType

from . import extraction_en, extraction_fr

BACKENDS: dict[str, ModuleType] = {"en": extraction_en, "fr": extraction_fr}


def get_backend(lang: str) -> ModuleType:
    try:
        return BACKENDS[lang]
    except KeyError:
        raise ValueError(f"unknown language: {lang!r} (expected one of {sorted(BACKENDS)})") from None
