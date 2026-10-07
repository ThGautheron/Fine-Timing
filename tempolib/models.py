"""The evaluated models: zero-shot ones from the Hub and our own fine-tuned runs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .utils import resolve


@dataclass(frozen=True)
class ModelSpec:

    key: str
    hub: Optional[str] = None
    run: Optional[str] = None
    max_length: int = 1024


MDEBERTA_HUB = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ROBERTA_HUB = "lytang/MiniCheck-RoBERTa-Large"

MODEL_LIST = [
    ModelSpec("mdeb_zs", hub=MDEBERTA_HUB, max_length=1024),
    ModelSpec("rob_zs", hub=ROBERTA_HUB, max_length=512),
    ModelSpec("mdeb_ft", run="mdeberta_main", max_length=1024),
    ModelSpec("rob_ft", run="roberta_main", max_length=512),
    ModelSpec("mdeb_rule_labels", run="mdeberta_rule_only", max_length=1024),
    ModelSpec("mdeb_generator_labels", run="mdeberta_generator_only", max_length=1024),
    ModelSpec("rob_rule_labels", run="roberta_rule_only", max_length=512),
    ModelSpec("rob_generator_labels", run="roberta_generator_only", max_length=512),
    ModelSpec("mdeb_ft_mnli", run="mdeberta_mnli", max_length=1024),
    ModelSpec("rob_ft_mnli", run="roberta_mnli", max_length=512),
    ModelSpec("mdeb_fr", run="mdeberta_fr", max_length=1024),
    ModelSpec("mdeb_fr_from_en", run="mdeberta_fr_from_en", max_length=1024),
]
MODELS = {m.key: m for m in MODEL_LIST}


def models_dir() -> Path:
    return resolve(os.environ.get("MODELS_DIR", "outputs/models"))


def checkpoint(spec: ModelSpec | str) -> Optional[str]:
    spec = MODELS[spec] if isinstance(spec, str) else spec
    if spec.hub:
        return spec.hub
    base = models_dir() / spec.run
    if (base / "final" / "config.json").exists():
        return str(base / "final")
    return str(base) if (base / "config.json").exists() else None
