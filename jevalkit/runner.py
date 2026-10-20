"""Streaming inference with verified resume, request sharding and explicit coverage."""

import fcntl
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .datasets import DecisionDataset
from .io import atomic_json, dumps, load_json, object_hash, sha256
from .metrics import grouped_report, score_predictions
from .schema import Unsupported, inference_request, is_capacity_error
from .security import is_credential_key


def selected_records(dataset, limit=None, rank=0, world_size=1):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("Require 0 <= rank < world_size")
    selected = 0
    for record in dataset:
        group = record["meta"].get("group_id") or record["index"]
        if int(object_hash(group), 16) % world_size != rank:
            continue
        yield record
        selected += 1
        if limit and selected >= limit:
            break


def read_journal(path, allowed):
    """Only recover a truncated final line; reject all other corruption."""
    records, end, seen = [], 0, set()
    if not path.exists():
        return records
    with path.open("rb") as stream:
        while line := stream.readline():
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    break
                raise ValueError("Corrupt prediction journal: " + str(path)) from None
            index = row["index"]
            if index not in allowed or index in seen:
                raise ValueError("Unknown or duplicate prediction record: " + str(index))
            records.append(row)
            seen.add(index)
            end = stream.tell()
    with path.open("r+b") as stream:
        stream.truncate(end)
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                stream.seek(0, 2)
                stream.write(b"\n")
    return records


def run(
    model_name,
    benchmark,
    output,
    *,
    root=None,
    model_options=None,
    resume=False,
    limit=None,
    rank=0,
    world_size=1,
    warmup=1,
    model_instance=None,
):
    from .config import build_model

    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    options = model_options or {}
    if any(is_credential_key(key) for key in options):
        raise ValueError("Pass a token_env variable name, never a credential value")
    dataset = DecisionDataset(benchmark, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another evaluator owns this output directory") from None
        records = list(selected_records(dataset, limit, rank, world_size))
        if not records:
            raise ValueError("No requests selected for this shard")
        # CLI accepts only token environment names; never token values.
        source_root = Path(__file__).parent
        signature = {
            "model": model_name,
            "model_options": options,
            "benchmark": benchmark,
            "dataset_sha256": dataset.manifest["sha256"],
            "protocol": dataset.manifest["protocol"],
            "selected_ids_sha256": object_hash([r["index"] for r in records]),
            "records": len(records),
            "limit": limit,
            "rank": rank,
            "world_size": world_size,
            "warmup": warmup,
            "jevalkit": __version__,
            "python": platform.python_version(),
            "source_sha256": {
                str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob("*.py"))
            },
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Output contains an existing run; use --resume or a new directory")
            if load_json(manifest_path)["signature"] != signature:
                raise ValueError("Resume data, code, model or evaluation settings changed")
        elif (output / "predictions.jsonl").exists():
            raise ValueError("Prediction journal has no run manifest")
        started = time.perf_counter()
        model = model_instance or build_model(model_name, root=root, **options)
        model.synchronize()
        load_seconds = time.perf_counter() - started
        runtime = model.metadata()
        if manifest_path.exists() and load_json(manifest_path)["runtime"] != runtime:
            model.close()
            raise ValueError("Resume model weights or runtime changed")
        manifest = {
            "signature": signature,
            "runtime": runtime,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest": dataset.manifest,
            "load_seconds": load_seconds,
            "timing_scope": "synchronized native request including preprocessing; excludes load/warmup/scoring",
        }
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)
        journal = output / "predictions.jsonl"
        predictions = read_journal(journal, {r["index"] for r in records})
        done = {r["index"] for r in predictions}
        try:
            for _ in range(warmup):
                for record in records[:1]:
                    try:
                        model.predict(
                            inference_request(record),
                            media_root=dataset.media_root,
                            assets=record.get("assets", []),
                        )
                    except Unsupported:
                        pass
            with journal.open("a", encoding="utf-8") as stream:
                for record in records:
                    if record["index"] in done:
                        continue
                    model.synchronize()
                    started = time.perf_counter()
                    prediction = {"index": record["index"], "status": "ok"}
                    try:
                        response = model.predict(
                            inference_request(record),
                            media_root=dataset.media_root,
                            assets=record.get("assets", []),
                        )
                        model.synchronize()
                        if set(response.get("answers", {})) != set(record["request"]["questions"]):
                            raise ValueError("Native response question IDs do not match the request")
                        dumps(response)  # Reject non-finite/non-JSON native values before journal append.
                        prediction["response"] = response
                    except Unsupported as exc:
                        prediction.update(status="unsupported", error=str(exc))
                    except Exception as exc:
                        prediction.update(
                            status="unsupported" if is_capacity_error(str(exc)) else "error",
                            exception=type(exc).__name__,
                            error=str(exc),
                        )
                    prediction["latency_s"] = time.perf_counter() - started
                    stream.write(dumps(prediction) + "\n")
                    stream.flush()
                    predictions.append(prediction)
                    if prediction.get("exception") in {"OutOfMemoryError", "AcceleratorError"}:
                        raise RuntimeError("GPU error recorded; stopping before reusing the device context")
                    if len(predictions) >= 5 and all(p["status"] == "error" for p in predictions[-5:]):
                        raise RuntimeError(
                            "Five consecutive request errors; inspect the native runtime before continuing"
                        )
                    if len(predictions) % 25 == 0:
                        print(
                            dumps(
                                {
                                    "event": "progress",
                                    "model": model_name,
                                    "benchmark": benchmark,
                                    "completed_requests": len(predictions),
                                    "planned": len(records),
                                }
                            ),
                            flush=True,
                        )
        finally:
            model.close()
        metrics = grouped_report(score_predictions(records, predictions))
        report = {
            "schema_version": 1,
            "model": model_name,
            "benchmark": benchmark,
            "dataset_sha256": dataset.manifest["sha256"],
            "protocol": dataset.manifest["protocol"],
            "attempted_requests": len(predictions),
            "selected_requests": len(records),
            "total_requests": dataset.manifest["records"],
            "rank": rank,
            "world_size": world_size,
            "complete": len(predictions) == dataset.manifest["records"] and world_size == 1,
            "smoke": limit is not None,
            "metrics": metrics,
            "runtime": runtime,
            "predictions_sha256": sha256(journal),
            "manifest_sha256": sha256(manifest_path),
        }
        from .datasets import official_report

        official = official_report(dataset, records, predictions, output)
        if official:
            report["official"] = official
        atomic_json(output / "metrics.json", report)
        print(
            dumps(
                {
                    "event": "complete",
                    "model": model_name,
                    "benchmark": benchmark,
                    "complete": report["complete"],
                    "metrics": metrics["overall"],
                }
            ),
            flush=True,
        )
        return report


def merge_shards(model_name, benchmark, shard_dirs, output, *, root=None):
    """Promote shards only after their exact request partition is proven complete."""
    from .io import jsonl

    dataset = DecisionDataset(benchmark, root)
    records = list(dataset)
    manifests, predictions, seen, ranks = [], [], set(), set()
    first = None
    for folder in map(Path, shard_dirs):
        report = load_json(folder / "metrics.json")
        manifest = load_json(folder / "manifest.json")
        signature = manifest["signature"]
        if report["model"] != model_name or report["benchmark"] != benchmark or report["smoke"]:
            raise ValueError("Shard model/benchmark mismatch or limited smoke shard")
        if report["dataset_sha256"] != dataset.manifest["sha256"]:
            raise ValueError("Shard dataset fingerprint differs from current benchmark")
        if (
            sha256(folder / "predictions.jsonl") != report["predictions_sha256"]
            or sha256(folder / "manifest.json") != report["manifest_sha256"]
        ):
            raise ValueError("Shard evidence changed")
        common = {k: v for k, v in signature.items() if k not in {"rank", "selected_ids_sha256", "records"}}
        common["runtime"] = manifest["runtime"]
        if first is None:
            first = common
        elif common != first:
            raise ValueError("Shard data/model/source/runtime/settings differ")
        rank, world = signature["rank"], signature["world_size"]
        if rank in ranks:
            raise ValueError("Duplicate shard rank")
        ranks.add(rank)
        expected = {r["index"] for r in selected_records(records, rank=rank, world_size=world)}
        rows = list(jsonl(folder / "predictions.jsonl"))
        actual = {r["index"] for r in rows}
        if len(rows) != len(actual) or actual != expected or seen & actual:
            raise ValueError("Shard request coverage mismatch or overlap")
        seen.update(actual)
        predictions.extend(rows)
        manifests.append({"path": str(folder.resolve()), "sha256": sha256(folder / "manifest.json")})
    if not first or ranks != set(range(first["world_size"])) or len(seen) != dataset.manifest["records"]:
        raise ValueError("Missing shards or incomplete global request coverage")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Merged output must be a new empty directory")
    positions = {r["index"]: i for i, r in enumerate(records)}
    predictions.sort(key=lambda r: positions[r["index"]])
    atomic_json(
        output / "manifest.json",
        {
            "signature": {**first, "operation": "verified-shard-merge"},
            "shards": manifests,
            "runtime": first["runtime"],
        },
    )
    journal = output / "predictions.jsonl"
    with journal.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(dumps(row) + "\n")
    report = {
        "schema_version": 1,
        "model": model_name,
        "benchmark": benchmark,
        "dataset_sha256": dataset.manifest["sha256"],
        "protocol": dataset.manifest["protocol"],
        "attempted_requests": len(predictions),
        "selected_requests": len(records),
        "total_requests": len(records),
        "rank": None,
        "world_size": first["world_size"],
        "complete": True,
        "smoke": False,
        "metrics": grouped_report(score_predictions(records, predictions)),
        "runtime": first["runtime"],
        "predictions_sha256": sha256(journal),
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    from .datasets import official_report

    official = official_report(dataset, records, predictions, output)
    if official:
        report["official"] = official
    atomic_json(output / "metrics.json", report)
    return report


def score_saved_run(folder, root=None):
    """Re-score stored native answers, including official metrics, without weights."""
    from .io import jsonl

    folder = Path(folder)
    with (folder / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Cannot re-score a run while inference is writing it") from None
        manifest_path = folder / "manifest.json"
        manifest = load_json(manifest_path)
        signature = manifest["signature"]
        dataset = DecisionDataset(signature["benchmark"], root)
        if dataset.manifest["sha256"] != signature["dataset_sha256"]:
            raise ValueError("Cannot re-score predictions against changed benchmark data")
        merged = signature.get("operation") == "verified-shard-merge"
        records = (
            list(dataset)
            if merged
            else list(
                selected_records(dataset, signature["limit"], signature["rank"], signature["world_size"])
            )
        )
        predictions = list(jsonl(folder / "predictions.jsonl"))
        seen = {p["index"] for p in predictions}
        if len(predictions) != len(seen) or seen != {r["index"] for r in records}:
            raise ValueError("Saved predictions do not cover the exact selected requests")
        report = {
            "schema_version": 1,
            "model": signature["model"],
            "benchmark": dataset.name,
            "dataset_sha256": dataset.manifest["sha256"],
            "protocol": dataset.manifest["protocol"],
            "attempted_requests": len(predictions),
            "selected_requests": len(records),
            "total_requests": dataset.manifest["records"],
            "complete": len(records) == dataset.manifest["records"]
            and (merged or signature["world_size"] == 1),
            "smoke": signature["limit"] is not None,
            "rank": signature.get("rank"),
            "world_size": signature["world_size"],
            "runtime": manifest["runtime"],
            "metrics": grouped_report(score_predictions(records, predictions)),
            "manifest_sha256": sha256(manifest_path),
            "predictions_sha256": sha256(folder / "predictions.jsonl"),
            "rescoring": {
                "version": __version__,
                "source_sha256": {
                    str(p.relative_to(Path(__file__).parent)): sha256(p)
                    for p in sorted(Path(__file__).parent.rglob("*.py"))
                },
            },
        }
        previous = folder / "metrics.json"
        if previous.exists() and "validation" in load_json(previous):
            report["validation"] = load_json(previous)["validation"]
        from .datasets import official_report

        official = official_report(dataset, records, predictions, folder)
        if official:
            report["official"] = official
        atomic_json(folder / "metrics.json", report)
        return report
