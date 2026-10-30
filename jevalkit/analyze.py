"""Find where target runs trail their own average and a reference group: slices, then single questions."""

import argparse
import statistics
from pathlib import Path

from .io import atomic_json, jsonl, load_json
from .metrics import SLICES


def load_run(folder):
    report = load_json(Path(folder) / "metrics.json")
    return {"name": report["model"], "folder": Path(folder), "benchmark": report["benchmark"],
            "metrics": report["metrics"]}


def check_runs(runs):
    if len({run["benchmark"] for run in runs}) != 1:
        raise ValueError("Compare runs of one benchmark at a time")
    if len({run["name"] for run in runs}) != len(runs):
        raise ValueError("Each run needs a distinct model name")


def analyze(targets, references, by=SLICES, min_questions=20, gap=0.05):
    """Slices where every target is gap below its own overall accuracy and below the reference median."""
    targets, references = [load_run(f) for f in targets], [load_run(f) for f in references]
    runs = targets + references
    check_runs(runs)
    tables, flagged = {}, []
    for name in by:
        if any(name not in run["metrics"]["groups"] for run in runs):
            raise ValueError(f"No {name} slices in metrics.json; re-score the runs with `jeval score`")
        rows = []
        for value, summary in targets[0]["metrics"]["groups"][name].items():
            accuracy = {run["name"]: run["metrics"]["groups"][name].get(value, {}).get("accuracy") for run in runs}
            known = [accuracy[r["name"]] for r in references if accuracy[r["name"]] is not None]
            row = {"slice": value, "hard_questions": summary["hard_questions"], "accuracy": accuracy,
                   "reference_median": statistics.median(known) if known else None}
            rows.append(row)
            below = [accuracy[t["name"]] is not None and t["metrics"]["overall"]["accuracy"] is not None
                     and accuracy[t["name"]] <= t["metrics"]["overall"]["accuracy"] - gap
                     and (row["reference_median"] is None or accuracy[t["name"]] <= row["reference_median"] - gap)
                     for t in targets]
            if summary["hard_questions"] >= min_questions and all(below):
                flagged.append({"by": name, **row})
        tables[name] = rows
    return {"benchmark": runs[0]["benchmark"], "targets": [t["name"] for t in targets],
            "references": [r["name"] for r in references], "slices": tables, "flagged": flagged}


def question_misses(targets, references, root=None, min_reference_share=0.75):
    """Hard questions every target answers wrong while most references answer right."""
    from .datasets import DecisionDataset
    from .metrics import score_predictions
    from .schema import target_label

    def correct(folder):
        report = load_json(Path(folder) / "metrics.json")
        rows = score_predictions(dataset, list(jsonl(Path(folder) / "predictions.jsonl")))
        return report["model"], {(r["index"], r["qid"]): r for r in rows if r["hard"]}

    check_runs([load_run(f) for f in [*targets, *references]])
    benchmark = load_json(Path(targets[0]) / "metrics.json")["benchmark"]
    dataset = list(DecisionDataset(benchmark, root))
    target_rows = dict(correct(f) for f in targets)
    reference_rows = dict(correct(f) for f in references)
    records = {r["index"]: r for r in dataset}
    misses = []
    for key in set.intersection(*(set(rows) for rows in target_rows.values())):
        if any(rows[key].get("correct") for rows in target_rows.values()):
            continue
        answered = [rows[key] for rows in reference_rows.values() if key in rows]
        share = sum(bool(r.get("correct")) for r in answered) / len(answered) if answered else 0
        if share >= min_reference_share:
            record, (index, qid) = records[key[0]], key
            misses.append({"index": index, "qid": qid, "reference_share": share,
                           "question": record["request"]["questions"][qid],
                           "expected": target_label(record["request"]["questions"][qid], record["targets"][qid]),
                           "predicted": {name: rows[key].get("predicted") for name, rows in target_rows.items()},
                           **{k: next(iter(target_rows.values()))[key].get(k) for k in SLICES}})
    misses.sort(key=lambda m: (-m["reference_share"], m["index"]))
    return misses


def markdown(result, misses=None):
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"
    names = result["targets"]
    lines = [f"# {', '.join(names)} on {result['benchmark']}", "",
             f"Reference group: {', '.join(result['references'])}.", "",
             "Flagged slices trail each target's own accuracy and the reference median.", ""]
    head = "| Slice | Questions | " + " | ".join(names) + " | Reference median |"
    rule = "| --- | ---: | " + " | ".join("---:" for _ in names) + " | ---: |"
    cells = lambda r: " | ".join(pct(r["accuracy"][n]) for n in names) + f" | {pct(r['reference_median'])} |"
    lines += [head, rule] + [f"| {r['by']}={r['slice']} | {r['hard_questions']} | " + cells(r) for r in result["flagged"]]
    for name, rows in result["slices"].items():
        lines += ["", f"## {name}", "", head, rule] + [f"| {r['slice']} | {r['hard_questions']} | " + cells(r) for r in rows]
    if misses is not None:
        lines += ["", f"## Questions every target misses ({len(misses)})", "",
                  "| Index | Question | Reference right | Target answer | Expected |", "| --- | --- | ---: | --- | --- |"]
        lines += [f"| {m['index']} | {m['question']['instructions'][:80].replace('|', '/')} | {pct(m['reference_share'])} | "
                  f"{'/'.join(str(v) for v in m['predicted'].values())} | {m['expected']} |" for m in misses]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jeval analyze", description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True, help="Run folders of the model under study")
    parser.add_argument("--references", nargs="+", required=True, help="Run folders of the reference group")
    parser.add_argument("--by", nargs="+", choices=SLICES, default=list(SLICES))
    parser.add_argument("--min-questions", type=int, default=20)
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--questions", action="store_true", help="Also list questions every target misses")
    parser.add_argument("--min-reference-share", type=float, default=0.75)
    parser.add_argument("--root")
    parser.add_argument("--output", help="Write analysis.json and analysis.md here")
    args = parser.parse_args(argv)
    result = analyze(args.targets, args.references, args.by, args.min_questions, args.gap)
    misses = question_misses(args.targets, args.references, args.root, args.min_reference_share) if args.questions else None
    text = markdown(result, misses)
    if args.output:
        atomic_json(Path(args.output) / "analysis.json", {**result, "question_misses": misses})
        (Path(args.output) / "analysis.md").write_text(text, encoding="utf-8")
    print(text, end="")
