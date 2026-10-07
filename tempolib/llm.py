"""Client for an OpenAI-compatible chat server (vLLM, LM Studio...) and parsing of its replies."""
from __future__ import annotations

import ast
import json
import re
import time
from typing import Optional


class LLMError(RuntimeError):
    """Raised when the server still fails after all the retries."""


class OpenAICompatibleClient:
    """A minimal chat client: one user prompt in, one reply out, with retries."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        seed: int = 13,
        timeout_s: int = 180,
        max_retries: int = 4,
    ):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries

    def complete(self, prompt: str, seed: int | None = None) -> dict:
        """Send `prompt` and return the reply (`content`), the served model and the attempt that worked."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    seed=self.seed if seed is None else seed,
                )
                return {
                    "content": resp.choices[0].message.content or "",
                    "model": getattr(resp, "model", self.model),
                    "attempt": attempt,
                }
            except Exception as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise LLMError(f"failed after {self.max_retries} attempts: {last_err}")


_QUOTED = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"', re.DOTALL)


def parse_variants(raw: str) -> list[str]:
    """The list of strings in a model reply, tolerating code fences and slightly broken JSON."""
    if not raw:
        return []
    txt = raw.strip()

    if "```" in txt:
        blocks = txt.split("```")
        txt = max(blocks, key=len)
        stripped = txt.lstrip()
        if stripped[:4].lower() == "json":
            txt = stripped[4:]

    i, j = txt.find("["), txt.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return []
    frag = txt[i: j + 1]

    for loader in (_load_strict, _load_relaxed, _load_regex):
        got = loader(frag)
        if got:
            return [s.strip() for s in got if isinstance(s, str) and s.strip()]
    return []


def _load_strict(frag: str) -> Optional[list]:
    try:
        v = json.loads(frag)
        return v if isinstance(v, list) else None
    except Exception:
        return None


def _load_relaxed(frag: str) -> Optional[list]:
    try:
        v = ast.literal_eval(frag)
        return list(v) if isinstance(v, (list, tuple)) else None
    except Exception:
        return None


def _load_regex(frag: str) -> Optional[list]:
    matches = _QUOTED.findall(frag)
    if not matches:
        return None
    out = []
    for m in matches:
        try:
            out.append(json.loads(f'"{m}"'))
        except Exception:
            out.append(m)
    return out
