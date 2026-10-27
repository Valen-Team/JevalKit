"""Verified native-image TSV conversion and upstream public vision scoring."""
import copy
import os
import shutil
from collections import Counter
from pathlib import Path

from ...io import dumps, sha256
from ...paths import data_root
from .._shared import commit_dataset
from ..decision_index import validate_official_response


def configure_model_options(model_name, options):
    from ...config import MODELS

    spec = MODELS.get(model_name)
    if spec and spec.family == "Valen":
        budget = options.setdefault("max_image_pixels", 1638400)
        if type(budget) is not int or budget != 1638400:
            raise ValueError("Decision Index public vision requires Valen's native image budget at 1638400 pixels")
    return options


def vision_record(row):
    if row["state"] != {}:
        raise ValueError("Unexpected nonempty official vision state; refusing to drop source content")
    images = row["images"]
    hashes = row["_evaluation"]["image_sha256"]
    if not images or len(images) != len(hashes):
        raise ValueError("Vision rows require ordered, checksummed images")
    return {
        "index": row["id"],
        "request": {
            "state": {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": path}} for path in images
            ]}]},
            "questions": copy.deepcopy(row["questions"]),
        },
        "targets": {key: {"label": value} for key, value in row["expected"].items()},
        "meta": {"subset": row["benchmark"], "domain": "vision", "modality": "image",
                 "group_id": row["id"], "source_set": row["set"], "subtask": row["subtask"],
                 "_evaluation": copy.deepcopy(row["_evaluation"]),
                 "official_state": copy.deepcopy(row["state"]), "official_prompt": copy.deepcopy(row["prompt"]),
                 "candidate_order": {key: list(q["criteria"]) for key, q in row["questions"].items()}},
        "assets": [{"path": path, "sha256": digest} for path, digest in zip(images, hashes)],
    }


def prepare(root=None, source=None, edition="0.4.1"):
    if edition != "0.4.1":
        raise ValueError("DecisionIndex_Vision supports board release 0.4.1")
    from decision_index.vision import spec
    from decision_index.vision.rows import read_rows, verify, write_rows

    root = data_root(root)
    folder = root / "raw" / "DecisionIndexVision-0.4.1"
    if source:
        if Path(source).resolve() != folder.resolve():
            shutil.copytree(source, folder, dirs_exist_ok=True)
    elif not (folder / "vision-rows.jsonl").exists():
        from decision_index.vision.build import build
        font = os.environ.get("DECISION_INDEX_FONT")
        if not font and Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf").is_file():
            font = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if not font:
            raise ValueError("Set DECISION_INDEX_FONT to Arial Bold before rebuilding Mind2Web, or provide --source with a verified suite")
        build(folder, font=font)
        write_rows(folder)
    checks = verify(folder)
    if (not checks["ok"] or set(checks["sets"]) != set(spec.SETS)
            or any("missing" in item for item in checks["sets"].values())
            or len(checks["benchmarks"]) != len(spec.BENCHMARKS)):
        raise ValueError("Official vision source, image or payload hashes do not match; inspect vision verify")
    rows = read_rows(folder / "vision-rows.jsonl")
    if len(rows) != 12210 or len({r["id"] for r in rows}) != 12210:
        raise ValueError("Expected the complete 12,210-row public vision suite")
    return commit_dataset("DecisionIndex_Vision", (vision_record(row) for row in rows), root,
        {"repository": "https://github.com/apolinario/decision-index", "edition": spec.EDITION,
         "leaderboard_version": edition, "score_scope": "public_vision", "checks": checks,
         "kit_revision": "25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d",
         "rows_sha256": sha256(folder / "vision-rows.jsonl"),
         "license": "Original dataset terms apply; local reconstruction, not a redistribution grant"},
        media_root="../raw/DecisionIndexVision-0.4.1", suite_dir="../raw/DecisionIndexVision-0.4.1",
        protocol="official-public-vision-0.4.1")


def evaluate(dataset, records, predictions, output):
    from decision_index.vision.score import score_run

    by_id = {p["index"]: p for p in predictions}
    path = Path(output) / "decision-index-vision-results.jsonl"
    counts = Counter()
    boundary_checks = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            prediction = by_id[record["index"]]
            row = {**record["meta"]["_evaluation"], "status": prediction["status"]}
            if row["status"] == "ok":
                row["response"] = prediction["response"]
                try:
                    boundary_checks += int(validate_official_response(record["request"]["questions"], row["response"]))
                except (KeyError, TypeError, ValueError):
                    row.update(status="error", error="Native response fails official vision validation")
                    row.pop("response")
            counts[row["status"]] += 1
            stream.write(dumps(row) + "\n")
    source = dataset.path.parent / dataset.manifest["suite_dir"]
    report = score_run(source / "vision-rows.jsonl", path, Path(output) / "official")
    report.update(leaderboard_version="0.4.1", public_suite_edition=report["edition"], score_scope="public_vision",
                  decision_index=report["public_vision_index"], completed=len(predictions), counts=dict(counts),
                  rounding_boundary_validation_checks=boundary_checks)
    return report
