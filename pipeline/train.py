"""Fine-tune a model as a binary verifier (faithful or not) on one of the runs of `configs/training.yaml`.

The epoch with the best dev accuracy is kept in `outputs/models/<run>/final`."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tempolib import load_config, read_jsonl, resolve, setup_console
from tempolib.hf import fp32_kwargs


def load_run(cfg: dict, name: str) -> dict:
    """Merge the common settings, the architecture and the overrides of one run of the training configuration."""
    if name not in cfg["runs"]:
        raise SystemExit(f"unknown run {name!r}; available: {', '.join(cfg['runs'])}")
    spec = dict(cfg["runs"][name])
    run = {**cfg["common"], **cfg["architectures"][spec.pop("arch")], **spec}
    base = resolve(cfg["models_dir"])
    run["name"] = name
    run["output_dir"] = str(base / name)
    if "init_from" in run:
        run["model_name"] = str(base / run.pop("init_from") / "final")
    run["dataset"] = str(resolve(run["dataset"]))
    return run


def train(run: dict) -> Path:
    """Fine-tune one run, keep the best epoch on dev, and return the folder of the saved model."""
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
        TrainingArguments, set_seed,
    )

    data_dir = Path(run["dataset"])
    out_dir = Path(run["output_dir"])
    set_seed(run["seed"])
    print("CUDA:", torch.cuda.is_available())

    train_fp, dev_fp = data_dir / "train.jsonl", data_dir / "dev.jsonl"
    if not train_fp.exists():
        raise SystemExit(f"missing: {train_fp} - build the datasets first")
    files = {"train": train_fp}
    if dev_fp.exists() and dev_fp.stat().st_size > 0:
        files["validation"] = dev_fp
    ds = DatasetDict({
        split: Dataset.from_list([{k: r[k] for k in ("document", "claim", "label")} for r in read_jsonl(fp)])
        for split, fp in files.items()})

    tok = AutoTokenizer.from_pretrained(run["model_name"], use_fast=False)

    def tokenize(batch):
        return tok(batch["document"], batch["claim"], truncation="only_first",
                   max_length=run["max_length"], return_token_type_ids=False)

    keep = {"input_ids", "attention_mask", "label"}
    ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in keep])
    has_dev = "validation" in ds and len(ds["validation"]) > 0
    if not has_dev:
        print("! dev set empty: training without evaluation or best-model selection.")

    model = AutoModelForSequenceClassification.from_pretrained(
        run["model_name"], num_labels=2, ignore_mismatched_sizes=True, **fp32_kwargs())

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": float((np.argmax(logits, axis=-1) == np.asarray(labels)).mean())}

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(run["learning_rate"]),
        warmup_ratio=run["warmup_ratio"],
        max_grad_norm=1.0,
        adam_epsilon=1e-6,
        per_device_train_batch_size=run["per_device_batch_size"],
        per_device_eval_batch_size=run["per_device_batch_size"],
        gradient_accumulation_steps=run["grad_accum"],
        num_train_epochs=run["epochs"],
        weight_decay=run["weight_decay"],
        eval_strategy="epoch" if has_dev else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_dev,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,
        logging_steps=50,
        seed=run["seed"],
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_dev else None,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
    )
    trainer.train()
    if has_dev:
        print("final dev evaluation:", trainer.evaluate())
    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    print(f"-> model saved: {final}")
    return final


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    ap.add_argument("--run", required=True, help="run name declared in the configuration")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    train(load_run(cfg, args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
