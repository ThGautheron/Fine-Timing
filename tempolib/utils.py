"""Small helpers: repository paths, configuration, and reading/writing (gzipped) text and JSON lines."""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def open_text(path: str | Path, mode: str = "r", newline: str | None = None):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8", newline=newline)
    return p.open(mode, encoding="utf-8", newline=newline)


def load_config(path: str | Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid configuration (mapping expected): {path}")
    return cfg


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def file_sha(path: str | Path, length: int = 12) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:length]


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
