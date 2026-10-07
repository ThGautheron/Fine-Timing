"""Shared code to build and evaluate the temporal-faithfulness datasets."""
__version__ = "2.0.0"

from .utils import load_config, now_iso, read_jsonl, resolve, setup_console, write_jsonl

__all__ = ["__version__", "load_config", "now_iso", "read_jsonl", "resolve", "setup_console", "write_jsonl"]
