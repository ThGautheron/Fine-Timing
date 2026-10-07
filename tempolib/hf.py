"""Hugging Face helpers: float32 loading and scoring of (document, claim) pairs."""
from __future__ import annotations

from typing import Callable, Optional


def fp32_kwargs() -> dict:
    import torch
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("4.56"):
        return {"dtype": torch.float32}
    return {"torch_dtype": torch.float32}


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def entailment_index(model) -> int:
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if len(id2label) == 2:
        return 1
    return next((i for i, lbl in id2label.items() if lbl == "entailment"), 1)


class Scorer:

    """P(faithful) of (document, claim) pairs under a sequence-classification checkpoint. The document is truncated first."""
    def __init__(self, model_path: str, max_length: int = 1024, batch_size: int = 1,
                 sort_key: Optional[Callable[[tuple], int]] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.device = device()
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **fp32_kwargs())
        self.entailment = entailment_index(self.model)
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.model.to(self.device).eval()

    def class_probabilities(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        import torch

        order = list(range(len(pairs)))
        if self.sort_key is not None:
            order.sort(key=lambda i: self.sort_key(pairs[i]))
        out: dict[int, list[float]] = {}
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first",
                           max_length=self.max_length, padding=True, return_token_type_ids=False,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).tolist()
            out.update(zip(idx, probs))
        return [out[i] for i in range(len(pairs))]

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [p[self.entailment] for p in self.class_probabilities(pairs)]

    def release(self) -> None:
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
