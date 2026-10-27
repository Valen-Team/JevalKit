"""Decision Index frozen-suite preparation and upstream official scoring."""

import copy
import hashlib
import shutil
from pathlib import Path
from ...io import dumps
from ...paths import data_root
from .._shared import commit_dataset


def install_digest_compat():
    """The pinned Decision Index kit uses Python 3.11's file_digest on Python 3.10."""
    if not hasattr(hashlib, "file_digest"):

        def file_digest(stream, algorithm):
            digest = hashlib.new(algorithm)
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            return digest

        hashlib.file_digest = file_digest


def prepare(root=None, source=None, edition="0.3", *, leaderboard_version=None):
    install_digest_compat()
    from decision_index import editions
    from decision_index.suite.io import Suite

    root = data_root(root)
    folder = root / "raw" / ("DecisionIndex-" + edition)
    folder.mkdir(parents=True, exist_ok=True)
    if source:
        for name in editions.files(edition):
            if (Path(source) / name).resolve() != (folder / name).resolve():
                shutil.copyfile(Path(source) / name, folder / name)
    else:
        from decision_index.suite.download import download

        download(folder, edition=edition)
    suite = Suite(folder, edition)
    checks = suite.verify(strict=True)
    if not checks.get("exclusions_match") or checks.get("subsets_match") is False:
        raise ValueError("Decision Index exclusion/subset checksums do not match the frozen release")

    def records():
        for row in suite.rows(apply_exclusions=True):
            questions = copy.deepcopy(row["questions"])
            for q in questions.values():
                if not isinstance(q["instructions"], str):
                    q["instructions"] = dumps(q["instructions"])
            yield {
                "index": row["_evaluation"]["run_id"],
                "request": {"state": row["state"], "questions": questions},
                "targets": {
                    k: {"label": v} if v is not None else {"unscored": True}
                    for k, v in row["expected"].items()
                },
                "meta": {
                    "subset": row["_evaluation"]["dataset"],
                    "domain": row.get("family", "unknown"),
                    "modality": "text",
                    "_evaluation": row["_evaluation"],
                    "expected": row["expected"],
                    "source_metadata": row.get("metadata", {}),
                },
                "assets": [],
            }

    dataset_name = "DecisionIndex_Text" if leaderboard_version else "DecisionIndex"
    source_metadata = {
        "repository": "https://github.com/apolinario/decision-index",
        "edition": edition,
        "checks": checks,
        "suite_manifest": suite.manifest(),
    }
    if leaderboard_version:
        source_metadata.update(leaderboard_version=leaderboard_version, public_suite_edition=edition,
                               score_scope="public_text", kit_revision="25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d")
    return commit_dataset(
        dataset_name,
        records(),
        root,
        source_metadata,
        protocol="official-public-text-" + leaderboard_version if leaderboard_version else "official-public-" + edition,
        suite_dir="../raw/DecisionIndex-" + edition,
    )


def validate_official_response(questions, response):
    """Keep the pinned validator's 1% boundary stable without changing scored answers."""
    from decision_index.engines import validate
    from ...schema import answer_probabilities

    try:
        validate(questions, response)
        return False
    except ValueError as error:
        if str(error) != "Incomplete/invalid probability distribution":
            raise
        validation_copy = copy.deepcopy(response)
        boundary = False
        for key, question in questions.items():
            if question["type"] != "choice":
                continue
            answer = validation_copy["answers"][key]
            normalized = answer_probabilities(question, answer)
            # Only floating-point noise beyond the declared boundary can pass.
            if abs(sum(answer["probabilities"].values()) - 1) > 0.01:
                answer["probabilities"] = normalized
                boundary = True
        if not boundary:
            raise
        validate(questions, validation_copy)
        return True


def evaluate(dataset, records, predictions, output):
    install_digest_compat()
    from decision_index.suite.io import Suite
    from decision_index.pipeline import score_run

    source = dataset.path.parent / dataset.manifest["suite_dir"]
    suite = Suite(source, dataset.manifest["source"]["edition"])
    by_id = {p["index"]: p for p in predictions}
    path = Path(output) / "decision-index-results.jsonl"
    rounding_boundary_checks = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            p = by_id[record["index"]]
            row = {
                **record["meta"]["_evaluation"],
                "status": p["status"],
                "total_wall_ms": p["latency_s"] * 1000,
                "model_request_wall_ms": p["latency_s"] * 1000,
            }
            if row["status"] == "refusal":
                row["status"] = "abstained"
            if "response" in p:
                row["response"] = p["response"]
                try:
                    rounding_boundary_checks += validate_official_response(record["request"]["questions"], row["response"])
                except (KeyError, ValueError, TypeError):
                    row.update(
                        status="error", error="Native response fails official Decision Index validation"
                    )
                    row.pop("response")
            if "error" in p:
                row["error"] = p["error"]
            stream.write(dumps(row) + "\n")
    report = score_run(suite, path, "jevalkit", Path(output) / "official")
    report["rounding_boundary_validation_checks"] = rounding_boundary_checks
    return report
