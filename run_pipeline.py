#!/usr/bin/env python3
"""Run the whole pipeline: anchors, splits, rewrites, datasets, training, evaluation, tables.

By default only the offline stages run (splits, datasets, evaluate, tables). They need no GPU and no LLM server: they use
the shipped data, the cached probabilities and the cached LLM verdicts. The other stages (anchors, generate, train,
illustration, retention) are added with --with or --all. For example:

    python run_pipeline.py --list
    python run_pipeline.py                         # default stages
    python run_pipeline.py --with train            # also fine-tune the models (GPU)
    python run_pipeline.py --all --own-rewrites    # everything, with freshly generated rewrites
    python run_pipeline.py --stages splits,datasets --dry-run"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

EN_CORPORA = ("dhq", "dm", "chdj")
TRAIN_RUNS = ["mdeberta_main", "roberta_main", "mdeberta_rule_only", "roberta_rule_only", "mdeberta_generator_only",
              "roberta_generator_only", "mdeberta_mnli", "roberta_mnli", "mdeberta_fr", "mdeberta_fr_from_en"]


@dataclass
class Step:
    cmd: list[str]
    skip_if: list[str] = field(default_factory=list)


@dataclass
class Stage:
    name: str
    description: str
    default: bool
    steps: list[Step]


def module(name: str, *args: str) -> list[str]:
    return [PY, "-m", name, *args]


def build_stages(args) -> list[Stage]:
    rewrites_en = "outputs/rewrites/en/candidates.jsonl" if args.own_rewrites else "data/rewrites/en.jsonl.gz"
    rewrites_fr = "outputs/rewrites/fr/candidates.jsonl" if args.own_rewrites else "data/rewrites/fr.jsonl.gz"
    en, fr = "outputs/work/en", "outputs/work/fr"
    datasets = "outputs/datasets"

    anchors = [Step(module("pipeline.build_anchors", "--input", f"data/rag_outputs/{c}.jsonl", "--name", c,
                           "--out-dir", "outputs/anchors", "--carry-annotations", f"data/anchors/{c}.csv.gz"),
                    [f"outputs/anchors/{c}.csv"]) for c in EN_CORPORA]

    splits = [
        Step(module("pipeline.prepare_splits", "--config", "configs/data_en.yaml"),
             ["outputs/splits/en/anchors_train.jsonl", "outputs/splits/en/anchors_test.jsonl"]),
        Step(module("pipeline.prepare_splits", "--config", "configs/data_fr.yaml"),
             ["outputs/splits/fr/anchors_train.jsonl", "outputs/splits/fr/anchors_test.jsonl"]),
    ]

    generate = [Step(module("pipeline.generate_rewrites", "--config", "configs/data_en.yaml"),
                     ["outputs/rewrites/en/candidates.jsonl"]),
                Step(module("pipeline.generate_rewrites", "--config", "configs/data_fr.yaml"),
                     ["outputs/rewrites/fr/candidates.jsonl"])]

    def rebuilt(step: Step) -> Step:
        return Step(step.cmd, [] if args.own_rewrites else step.skip_if)

    dataset_steps = [
        Step(module("pipeline.judge_rewrites", rewrites_en, "-o", f"{en}/judged.jsonl"), [f"{en}/judged.jsonl"]),
        Step(module("pipeline.build_labelled", f"{en}/judged.jsonl", "-o", f"{en}/labelled_rule.jsonl",
                    "--label-source", "rule"), [f"{en}/labelled_rule.jsonl"]),
        Step(module("pipeline.assemble_dataset", "--config", "configs/data_en.yaml", "--labelled",
                    f"{en}/labelled_rule.jsonl", "--out", f"{en}/preassemble"), [f"{en}/preassemble/train.jsonl"]),
        Step(module("pipeline.rebalance_dataset", "--in", f"{en}/preassemble", "--out", f"{en}/rebalanced"),
             [f"{en}/rebalanced/train.jsonl"]),
        Step(module("pipeline.relabel_precision", "--in", f"{en}/rebalanced", "--out", f"{en}/relabelled"),
             [f"{en}/relabelled/train.jsonl"]),
        Step(module("pipeline.add_silver_pairs", "--in", f"{en}/relabelled", "--out", f"{datasets}/en_main"),
             [f"{datasets}/en_main/train.jsonl"]),
        Step(module("pipeline.build_labelled", rewrites_en, "-o", f"{en}/labelled_generator_only.jsonl",
                    "--label-source", "llm", "--keep-all", "--drop-conflicts"),
             [f"{en}/labelled_generator_only.jsonl"]),
        Step(module("pipeline.assemble_dataset", "--config", "configs/data_en.yaml", "--labelled",
                    f"{en}/labelled_generator_only.jsonl", "--out", f"{datasets}/en_generator_only", "--no-human"),
             [f"{datasets}/en_generator_only/train.jsonl"]),
        Step(module("pipeline.build_labelled", f"{en}/judged.jsonl", "-o", f"{en}/labelled_rule_only.jsonl",
                    "--label-source", "rule", "--drop-conflicts"), [f"{en}/labelled_rule_only.jsonl"]),
        Step(module("pipeline.assemble_dataset", "--config", "configs/data_en.yaml", "--labelled",
                    f"{en}/labelled_rule_only.jsonl", "--out", f"{datasets}/en_rule_only", "--no-human"),
             [f"{datasets}/en_rule_only/train.jsonl"]),
        Step(module("pipeline.add_mnli", "--src", f"{datasets}/en_main", "--out", f"{datasets}/en_mnli"),
             [f"{datasets}/en_mnli/train.jsonl"]),
        Step(module("pipeline.judge_rewrites", rewrites_fr, "-o", f"{fr}/judged.jsonl", "--lang", "fr"),
             [f"{fr}/judged.jsonl"]),
        Step(module("pipeline.build_labelled", f"{fr}/judged.jsonl", "-o", f"{fr}/labelled_rule.jsonl",
                    "--label-source", "rule"), [f"{fr}/labelled_rule.jsonl"]),
        Step(module("pipeline.assemble_dataset", "--config", "configs/data_fr.yaml", "--labelled",
                    f"{fr}/labelled_rule.jsonl", "--out", f"{fr}/preassemble"), [f"{fr}/preassemble/train.jsonl"]),
        Step(module("pipeline.relabel_precision", "--in", f"{fr}/preassemble", "--out", f"{datasets}/fr", "--lang", "fr"),
             [f"{datasets}/fr/train.jsonl"]),
    ]
    dataset_steps = [rebuilt(s) for s in dataset_steps]

    runs = args.runs.split(",") if args.runs else TRAIN_RUNS
    train = [Step(module("pipeline.train", "--run", r), [f"outputs/models/{r}/final/config.json"]) for r in runs]
    llm_args = ["--base-url", args.llm_base_url] if args.llm_base_url else []
    evaluate = [Step(module("analysis.score_models")), Step(module("analysis.llm_judge", *llm_args))]
    tables = [Step(module("analysis.table5")), Step(module("analysis.dataset_statistics")),
              Step(module("analysis.persistent_failures"))]
    illustration = [Step(module("analysis.illustration", *llm_args))]
    retention = [Step(module(f"analysis.{s}")) for s in ("retention_nli", "retention_aggrefact")]

    return [
        Stage("anchors", "re-extract the anchor tables from the RAG outputs (needs the bge-m3 model; annotated tables are shipped)", False, anchors),
        Stage("splits", "split the annotated anchors by document into train / dev / test", True, splits),
        Stage("generate", "LLM rewrites of the faithful anchors (needs an OpenAI-compatible server; rewrites are shipped)", False, generate),
        Stage("datasets", "judge the rewrites, label, assemble, rebalance and build every training set", True, dataset_steps),
        Stage("train", "fine-tune the models (GPU; hours)", False, train),
        Stage("evaluate", "score the models (checkpoints or cached probabilities) and the LLM judges (cached verdicts)", True, evaluate),
        Stage("tables", "Table 5, data statistics (Tables 2 and 4, Sect. 4) and persistent failures (Sect. 6.1)", True, tables),
        Stage("illustration", "Table 3: one source sentence, seven rewrites (downloads the zero-shot models; LLM verdicts cached)", False, illustration),
        Stage("retention", "Sect. 6.2: retention of general NLI and fact checking after fine-tuning (GPU, Hugging Face datasets)", False, retention),
    ]


def run_step(step: Step, args) -> bool:
    if step.skip_if and not args.force and not args.dry_run and all((ROOT / p).exists() for p in step.skip_if):
        print(f"  = up to date: {' '.join(step.cmd[1:])}")
        return True
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print("  $ " + " ".join(["python", *step.cmd[1:]]), flush=True)
    if args.dry_run:
        return True
    code = subprocess.run(step.cmd, cwd=ROOT, env=env).returncode
    if code != 0:
        print(f"  ! failed with exit code {code}", file=sys.stderr)
    return code == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--stages", help="comma-separated stages to run (default: the default stages)")
    ap.add_argument("--with", dest="extra", help="comma-separated optional stages added to the default ones")
    ap.add_argument("--skip", help="comma-separated stages to skip")
    ap.add_argument("--all", action="store_true", help="every stage")
    ap.add_argument("--runs", help="comma-separated training runs (default: all of configs/training.yaml)")
    ap.add_argument("--own-rewrites", action="store_true",
                    help="build the datasets from outputs/rewrites/ (stage generate) instead of data/rewrites/")
    ap.add_argument("--llm-base-url", help="OpenAI-compatible server of the LLM judges (cached verdicts need none)")
    ap.add_argument("--force", action="store_true", help="rerun the steps whose outputs already exist")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    stages = build_stages(args)
    by_name = {s.name: s for s in stages}
    if args.list:
        for s in stages:
            print(f"{s.name:10} {'[default]' if s.default else '[optional]':11} {s.description}")
        return 0

    if args.stages:
        selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    elif args.all:
        selected = [s.name for s in stages]
    else:
        selected = [s.name for s in stages if s.default]
        selected += [s.strip() for s in (args.extra or "").split(",") if s.strip() and s.strip() not in selected]
    skipped = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    unknown = [s for s in selected + sorted(skipped) if s not in by_name]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}; use --list", file=sys.stderr)
        return 2
    order = [s.name for s in stages if s.name in selected and s.name not in skipped]

    t0 = time.time()
    failed = []
    for name in order:
        stage = by_name[name]
        print(f"\n== {name}: {stage.description}", flush=True)
        for step in stage.steps:
            if not run_step(step, args):
                failed.append(" ".join(step.cmd[1:]))
                if not args.keep_going:
                    print(f"\npipeline stopped at stage '{name}'", file=sys.stderr)
                    return 1
    print(f"\n{'dry run of' if args.dry_run else 'done:'} {', '.join(order)} ({time.time() - t0:.0f} s)")
    if failed:
        print("failed steps:\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
