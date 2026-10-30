"""jeval workflow list | run | score."""

import argparse

from ..config import MODELS
from ..io import dumps
from . import WORKFLOWS


def main(argv=None):
    from ..cli import options

    parser = argparse.ArgumentParser(prog="jeval workflow", description="End-to-end workflows with decisions in code")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List workflows")
    run = sub.add_parser("run", help="Run a workflow with one model")
    run.add_argument("workflow", choices=WORKFLOWS)
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--output", required=True)
    run.add_argument("--option", action="append", default=[])
    run.add_argument("--root")
    run.add_argument("--limit", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--workers", type=int, default=1, help="Concurrent cases; keep 1 for local GPU models")
    run.add_argument("--dataset-revision")
    compare = sub.add_parser("compare", help="Agreement, cost and time of saved runs and published runs")
    compare.add_argument("workflow", choices=WORKFLOWS)
    compare.add_argument("runs", nargs="*")
    compare.add_argument("--reference")
    compare.add_argument("--root")
    score = sub.add_parser("score", help="Re-score a saved workflow run")
    score.add_argument("run_dir")
    score.add_argument("--root")
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(WORKFLOWS))
        return
    if args.command == "compare":
        from .runner import compare

        print(compare(args.workflow, args.runs, args.reference, args.root), end="")
        return
    if args.command == "run":
        from .runner import run_workflow

        report = run_workflow(args.workflow, args.model, args.output, root=args.root,
                              model_options=options(args.option), limit=args.limit, resume=args.resume,
                              workers=args.workers, dataset_revision=args.dataset_revision)
    else:
        from .runner import score_run

        report = score_run(args.run_dir, root=args.root)
    reference = report["references"][report["default_reference"]]
    print(dumps({"workflow": report["workflow"], "model": report["model"], "cases": report["cases"],
                 "status": report["status"], "reference": report["default_reference"],
                 "exact_actions": reference["exact_actions"], "primary_action": reference["primary_action"],
                 "mean_cost_usd": report["mean_cost_usd"], "mean_wall_time_s": report["mean_wall_time_s"]}))
