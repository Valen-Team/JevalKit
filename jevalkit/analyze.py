"""Compare saved runs slice by slice to find where a model trails its own average and the other runs."""

import argparse
from pathlib import Path

from .io import atomic_json, load_json
from .metrics import SLICES


def load_run(folder):
    report = load_json(Path(folder) / "metrics.json")
    runtime = report.get("runtime") or {}
    name = report["model"] + (f" ({runtime['model']})" if runtime.get("model") and "/" not in runtime["model"] else "")
    return {"name": name, "benchmark": report["benchmark"], "metrics": report["metrics"]}


def analyze(folders, by=SLICES, min_questions=20, gap=0.05):
    """Slices where the first run's accuracy is gap below its overall accuracy and below the best other run."""
    runs = [load_run(folder) for folder in folders]
    if len({run["benchmark"] for run in runs}) != 1:
        raise ValueError("Compare runs of one benchmark at a time")
    target, others = runs[0], runs[1:]
    overall = target["metrics"]["overall"]["accuracy"]
    tables, flagged = {}, []
    for name in by:
        groups = target["metrics"]["groups"].get(name)
        if groups is None:
            raise ValueError(f"No {name} slices in metrics.json; re-score the run with `jeval score`")
        rows = []
        for value, summary in groups.items():
            row = {"slice": value, "hard_questions": summary["hard_questions"],
                   "accuracy": {run["name"]: run["metrics"]["groups"].get(name, {}).get(value, {}).get("accuracy")
                                for run in runs},
                   "ece": summary["ece"], "coverage": summary["coverage"]}
            rows.append(row)
            mine = summary["accuracy"]
            best = max((v for k, v in row["accuracy"].items() if k != target["name"] and v is not None), default=None)
            if (mine is not None and summary["hard_questions"] >= min_questions and overall is not None
                    and mine <= overall - gap and (best is None or mine <= best - gap)):
                flagged.append({"by": name, **row, "below_overall": overall - mine,
                                "below_best_other": None if best is None else best - mine})
        tables[name] = rows
    flagged.sort(key=lambda r: -r["below_overall"])
    return {"benchmark": target["benchmark"], "target": target["name"], "overall_accuracy": overall,
            "runs": [run["name"] for run in runs], "slices": tables, "flagged": flagged}


def markdown(result):
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"
    lines = [f"# {result['target']} on {result['benchmark']}", "",
             f"Overall accuracy {pct(result['overall_accuracy'])}. Flagged slices trail it and every other run.", ""]
    if result["flagged"]:
        lines += ["| Slice | Questions | " + " | ".join(result["runs"]) + " |",
                  "| --- | ---: | " + " | ".join("---:" for _ in result["runs"]) + " |"]
        lines += [f"| {r['by']}={r['slice']} | {r['hard_questions']} | "
                  + " | ".join(pct(r["accuracy"][run]) for run in result["runs"]) + " |" for r in result["flagged"]]
    else:
        lines.append("No slice is flagged.")
    for name, rows in result["slices"].items():
        lines += ["", f"## {name}", "", "| Slice | Questions | " + " | ".join(result["runs"]) + " | ECE |",
                  "| --- | ---: | " + " | ".join("---:" for _ in result["runs"]) + " | ---: |"]
        lines += [f"| {r['slice']} | {r['hard_questions']} | "
                  + " | ".join(pct(r["accuracy"][run]) for run in result["runs"]) + f" | {pct(r['ece'])} |"
                  for r in rows]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jeval analyze", description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run folders of one benchmark; the first is analyzed")
    parser.add_argument("--by", nargs="+", choices=SLICES, default=list(SLICES))
    parser.add_argument("--min-questions", type=int, default=20)
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--output", help="Write analysis.json and analysis.md here")
    args = parser.parse_args(argv)
    result = analyze(args.runs, args.by, args.min_questions, args.gap)
    text = markdown(result)
    if args.output:
        atomic_json(Path(args.output) / "analysis.json", result)
        (Path(args.output) / "analysis.md").write_text(text, encoding="utf-8")
    print(text, end="")
