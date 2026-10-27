"""Generate auditable local results; never fill missing scores from model cards."""
import csv
from pathlib import Path
from .datasets import BENCHMARKS
from .io import atomic_json, load_json, sha256


def primary_results(rows):
    """Use the latest complete measurement, without selecting the best repeat."""
    latest = {}
    for row in rows:
        if not row.get("input_conformant", True):
            continue
        unsupported = row["complete"] and not row["smoke"] and row["coverage"] == 0 and row["unsupported"] > 0 and not row["failed"]
        if not row["eligible"] and not unsupported:
            continue
        key = (row["model"], row["benchmark"], row["dataset_sha256"], row["protocol"])
        if key not in latest or (row["created_utc"], row["evidence"]) > (latest[key]["created_utc"], latest[key]["evidence"]):
            latest[key] = row
    return list(latest.values())


def display_score(row):
    if row["coverage"] == 0:
        return "—"
    if row["public_decision_index"] is not None:
        return f'{row["public_decision_index"]:.2f}'
    return "—" if row["accuracy"] is None else f'{row["accuracy"] * 100:.2f}%'


def build_leaderboard(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for source in inputs:
        source = Path(source)
        paths = [source] if source.is_file() else sorted(source.rglob("metrics.json"))
        for path in paths:
            report = load_json(path)
            if report.get("schema_version") != 1 or "dataset_sha256" not in report:
                continue
            if report.get("benchmark") not in BENCHMARKS:
                continue
            manifest = path.parent / "manifest.json"
            predictions = path.parent / "predictions.jsonl"
            if not manifest.exists() or not predictions.exists():
                raise ValueError("Result evidence is missing: " + str(path))
            if report["manifest_sha256"] != sha256(manifest) or report["predictions_sha256"] != sha256(predictions):
                raise ValueError("Result evidence changed: " + str(path))
            key = (report["model"], report["benchmark"], report["dataset_sha256"], report["protocol"], str(path.resolve()))
            if key in seen:
                continue
            seen.add(key)
            metrics = report["metrics"]["overall"]
            created = load_json(manifest).get("created_utc", "")
            input_conformant = report.get("validation", {}).get("inference_input_conformant", True)
            official = report.get("official", {})
            official_valid = official.get("complete", True) and not official.get("counts", {}).get("error", 0)
            eligible = report["complete"] and not report["smoke"] and metrics["failed"] == 0 and metrics["valid"] > 0 and input_conformant and official_valid
            rows.append({"model": report["model"], "benchmark": report["benchmark"],
                         "protocol": report["protocol"], "dataset_sha256": report["dataset_sha256"],
                         "complete": report["complete"], "smoke": report["smoke"], "eligible": eligible,
                         "input_conformant": input_conformant,
                         "questions": metrics["questions"], "coverage": metrics["coverage"],
                         "accuracy": official.get("accuracy", metrics["accuracy"]),
                         "brier": metrics["brier"], "nll": metrics["nll"], "ece": metrics["ece"],
                         "public_decision_index": official.get("decision_index"),
                         "failed": metrics["failed"], "unsupported": metrics["unsupported"],
                         "latency_s": metrics["request_latency_mean_s"],
                         "hardware": report["runtime"].get("hardware", report["runtime"].get("device", "unknown")),
                         "origin": report["runtime"].get("origin", "jevalkit-inference"),
                         "created_utc": created,
                         "evidence": str(path.resolve())})
    rows.sort(key=lambda r: (r["benchmark"], r["dataset_sha256"], r["protocol"], not r["eligible"],
                            -(r["public_decision_index"] if r["public_decision_index"] is not None else (r["accuracy"] or 0) * 100), r["model"]))
    primary = primary_results(rows)
    atomic_json(output / "leaderboard.json", {"schema_version": 1, "results": rows, "primary_results": primary,
                "policy": "Only complete non-smoke runs without runtime failures are eligible. Compare identical dataset hashes and protocols; latency also requires matching hardware. Public Decision Index excludes private components."})
    if rows:
        with (output / "leaderboard.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        (output / "leaderboard.tsv").write_text("", encoding="utf-8")
    lines = ["# JevalKit local leaderboard", "",
             "The overview uses the latest complete run for each model, benchmark, data hash and protocol; repeated runs are never selected by their best score.",
             "Missing and fully unsupported evaluations have no score. The detailed table retains every run, including smoke tests and imported measurements.", ""]
    lines += ["Only currently registered Jev-related benchmarks are included; retired tasks remain in the original result storage.", ""]
    columns = sorted({(r["benchmark"], r["dataset_sha256"], r["protocol"]) for r in primary})
    labels = [name if sum(c[0] == name for c in columns) == 1 else f"{name} [{digest[:8]} / {protocol}]"
              for name, digest, protocol in columns]
    if columns:
        lines += ["| Model | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(columns)]
        lookup = {(r["model"], r["benchmark"], r["dataset_sha256"], r["protocol"]): r for r in primary}
        for model in sorted({r["model"] for r in primary}):
            cells = []
            for column in columns:
                row = lookup.get((model, *column))
                score = display_score(row) if row else "—"
                if row and row["origin"] == "imported-upstream-measurement":
                    score += " (imported)"
                cells.append(score)
            lines.append("| " + model + " | " + " | ".join(cells) + " |")
        lines += ["", "Accuracy is shown as a percentage. DecisionIndex uses the upstream weighted public index (0–100).", ""]
    lines += ["| Model | Benchmark | Score | Metric | Valid coverage | Status | Questions | Evidence |",
              "|---|---|---:|---|---:|---|---:|---|"]
    for row in rows:
        status = "full" if row["eligible"] else "smoke" if row["smoke"] else "partial/errors"
        unsupported = row["complete"] and row["coverage"] == 0 and row["unsupported"] > 0 and row["failed"] == 0
        if unsupported:
            status = "unsupported"
        if not row["input_conformant"]:
            status = "input-contract error"
        if row["origin"] == "imported-upstream-measurement":
            status = "imported " + status
        accuracy = "—" if row["accuracy"] is None else f'{row["accuracy"] * 100:.2f}%'
        score = f'{row["public_decision_index"]:.2f}' if row["public_decision_index"] is not None else accuracy
        if unsupported:
            score = "—"
        metric = "public index" if row["public_decision_index"] is not None else "accuracy"
        lines.append(f'| {row["model"]} | {row["benchmark"]} | {score} | {metric} | {row["coverage"] * 100:.2f}% | {status} | {row["questions"]} | [run](<{row["evidence"]}>) |')
    lines += ["", "TSV/JSON include dataset hashes, protocol, hardware and evidence paths.",
              "Accuracy counts failed/unsupported hard questions as incorrect. Probability metrics use valid outputs.",
              "Decision Index is the upstream public index; its private Full score is not inferred."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
