"""Command line entry point, including zero-GPU dataset verification."""

import argparse
import json
from pathlib import Path
from .config import MODELS, model_catalog
from .datasets import BENCHMARKS, DecisionDataset
from .io import dumps
from .security import is_credential_key


def options(items):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("Options must be key=value")
        key, value = item.split("=", 1)
        if is_credential_key(key):
            raise ValueError("Use token_env=VARIABLE_NAME instead of a credential value")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jeval", description="Reproducible typed decision model evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("list", help="List implemented adapters and benchmarks")
    catalog.add_argument("kind", choices=("models", "benchmarks"))
    catalog.add_argument("--json", action="store_true")
    prepare = sub.add_parser("prepare", help="Download/import public data into retained TSVs")
    prepare.add_argument("benchmarks", nargs="+", choices=BENCHMARKS)
    prepare.add_argument("--root")
    prepare.add_argument("--source", help="Existing native benchmark directory")
    prepare.add_argument("--edition", default="0.3", choices=("0.2.1", "0.3"))
    verify = sub.add_parser("verify", help="Verify TSV schemas, counts, hashes, optionally all media")
    verify.add_argument("benchmarks", nargs="+", choices=BENCHMARKS)
    verify.add_argument("--root")
    verify.add_argument("--media", action="store_true")
    evaluate = sub.add_parser("run", help="Evaluate a model on one or more benchmarks")
    evaluate.add_argument("--model", required=True, choices=MODELS)
    evaluate.add_argument("--data", required=True, nargs="+", choices=BENCHMARKS)
    evaluate.add_argument("--root")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--option", action="append", default=[])
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--rank", type=int, default=0)
    evaluate.add_argument("--world-size", type=int, default=1)
    evaluate.add_argument("--warmup", type=int, default=1)
    board = sub.add_parser("leaderboard", help="Aggregate verified prediction/metric artifacts")
    board.add_argument("inputs", nargs="+")
    board.add_argument("--output", required=True)
    merge = sub.add_parser("merge", help="Verify and merge complete request shards")
    merge.add_argument("shards", nargs="+")
    merge.add_argument("--model", required=True, choices=MODELS)
    merge.add_argument("--data", required=True, choices=BENCHMARKS)
    merge.add_argument("--root")
    merge.add_argument("--output", required=True)
    score = sub.add_parser("score", help="Re-score stored predictions without loading a model")
    score.add_argument("run_dir")
    score.add_argument("--root")
    check = sub.add_parser("check", help="Check typed response/JSON input contracts without benchmark scores")
    check.add_argument("--model", required=True, choices=MODELS)
    check.add_argument("--root")
    check.add_argument("--option", action="append", default=[])
    check.add_argument("--output", required=True)
    imported = sub.add_parser(
        "import-decision-index", help="Verify and re-score actual upstream prediction artifacts"
    )
    imported.add_argument("--model", required=True, choices=MODELS)
    imported.add_argument("--results", required=True)
    imported.add_argument("--environment", required=True, nargs="+")
    imported.add_argument("--output", required=True)
    imported.add_argument("--root")
    matrix = sub.add_parser("matrix", help="Evaluate a multi-model matrix with isolated sequential runtimes")
    matrix.add_argument("config")
    matrix.add_argument("--root")
    matrix.add_argument("--status")
    args = parser.parse_args(argv)
    if args.command == "list":
        data = model_catalog() if args.kind == "models" else BENCHMARKS
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            for name, value in data.items():
                print(
                    name + "\t" + value["family"] + "\t" + value.get("transport", value.get("modality", ""))
                )
    elif args.command == "prepare":
        from .datasets import prepare_dataset

        if args.source and len(args.benchmarks) != 1:
            parser.error("--source accepts one benchmark at a time")
        for name in args.benchmarks:
            result = prepare_dataset(name, args.root, args.source, args.edition)
            print(dumps(result), flush=True)
    elif args.command == "verify":
        for name in args.benchmarks:
            print(dumps(DecisionDataset(name, args.root).verify(args.media)), flush=True)
    elif args.command == "run":
        from .runner import run

        for name in args.data:
            destination = Path(args.output) / args.model / name
            if args.world_size > 1:
                destination /= f"shard-{args.rank:03d}"
            run(
                args.model,
                name,
                destination,
                root=args.root,
                model_options=options(args.option),
                resume=args.resume,
                limit=args.limit,
                rank=args.rank,
                world_size=args.world_size,
                warmup=args.warmup,
            )
    elif args.command == "leaderboard":
        from .leaderboard import build_leaderboard

        rows = build_leaderboard(args.inputs, args.output)
        print(dumps({"results": len(rows), "output": args.output}))
    elif args.command == "merge":
        from .runner import merge_shards

        report = merge_shards(args.model, args.data, args.shards, args.output, root=args.root)
        print(dumps({"model": args.model, "benchmark": args.data, "complete": report["complete"]}))
    elif args.command == "score":
        from .runner import score_saved_run

        report = score_saved_run(args.run_dir, args.root)
        print(
            dumps(
                {
                    "model": report["model"],
                    "benchmark": report["benchmark"],
                    "complete": report["complete"],
                    "metrics": report["metrics"]["overall"],
                }
            )
        )
    elif args.command == "import-decision-index":
        from .import_results import import_decision_index

        report = import_decision_index(
            args.model, args.results, args.environment, args.output, root=args.root
        )
        print(
            dumps(
                {
                    "model": report["model"],
                    "complete": report["complete"],
                    "public_decision_index": report["official"]["decision_index"],
                }
            )
        )
    elif args.command == "matrix":
        from .matrix import run_matrix

        report = run_matrix(args.config, args.root, args.status)
        if not report["complete"]:
            raise SystemExit(1)
    elif args.command == "check":
        from .conformance import check_model

        report = check_model(args.model, args.output, args.root, options(args.option))
        print(dumps({"model": args.model, "passed": report["passed"], "output": args.output}))
        if not report["passed"]:
            raise SystemExit(1)
