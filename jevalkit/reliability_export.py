"""Convert retained JevalKit runs to the offline reliability protocol without inference."""
import argparse
import json
from pathlib import Path

from .datasets import DecisionDataset
from .metrics import score_predictions
from .io import object_hash, sha256
from .runner import selected_records
from .schema import candidates, target_probabilities, target_label
from .reliability import digest, write_json, validate_cache


def runtime_identity(runtime, protocol):
    name = Path(runtime["model"]).name
    model = "Valen-Team/" + name if name in ("Valen-0.8B", "Valen-2B", "Valen-4B") else runtime["model"]
    artifacts = runtime.get("artifacts_sha256") or {
        **runtime.get("weights", {}), **runtime.get("runtime_sha256", {}),
        "config.json": runtime["config_sha256"]}
    return {"model": model, "revision": runtime["revision"],
            "backend": ":".join(("transformers", runtime["execution"], runtime["attention"],
                                   "cuda" if str(runtime.get("device", runtime.get("hardware", ""))).startswith("cuda") or "NVIDIA" in runtime.get("hardware", "") else "cpu")),
            "precision": runtime["dtype"], "artifacts_sha256": artifacts,
            "protocol_sha256": digest(protocol)}


def export_run(run_dir, output, role, protocol, group_namespace, root=None, group_by="meta"):
    if group_by not in ("meta", "normalized-state"):
        raise ValueError("Unknown grouping policy")
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["signature"]["model"] not in ("Valen-0.8B", "Valen-2B", "Valen-4B"):
        raise ValueError("This exporter currently binds native Valen runs only")
    runtime = manifest["runtime"]
    if runtime.get("temperature") != 1.0 or runtime.get("calibration_sha256"):
        raise ValueError("Export raw T=1 runs; replay applies calibration exactly once")
    dataset = DecisionDataset(manifest["signature"]["benchmark"], root)
    if dataset.manifest["sha256"] != manifest["signature"]["dataset_sha256"]:
        raise ValueError("Dataset differs from retained run")
    prediction_hash, manifest_hash = sha256(run_dir / "predictions.jsonl"), sha256(run_dir / "manifest.json")
    if (run_dir / "metrics.json").exists():
        original = json.loads((run_dir / "metrics.json").read_text())
        if (original.get("predictions_sha256") != prediction_hash or original.get("manifest_sha256") != manifest_hash):
            raise ValueError("Retained run integrity mismatch against original metrics hashes")
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    by_id = {row["index"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    signature = manifest["signature"]
    records = (list(dataset) if signature.get("operation") == "verified-shard-merge" else
               list(selected_records(dataset, signature.get("limit"), signature.get("rank", 0), signature.get("world_size", 1))))
    if (set(by_id) != {r["index"] for r in records} or len(records) != signature["records"]
            or signature.get("selected_ids_sha256", object_hash([r["index"] for r in records])) != object_hash([r["index"] for r in records])):
        raise ValueError("Incomplete or unknown prediction IDs")
    scored_rows = {(r["index"], r["qid"]): r for r in score_predictions(records, predictions)}
    rows = []
    for record in records:
        # An explicit source-entity group is mandatory. Request index cannot detect
        # paraphrases/related media spread across separate records.
        group = record["meta"].get("group_id")
        if group_by == "normalized-state":
            state = record["request"]["state"]
            if not isinstance(state, str):
                raise ValueError("normalized-state grouping supports text only")
            group = digest(" ".join(state.casefold().split()))
        if group is None:
            raise ValueError("Source group_id missing; annotate entity/media groups before export")
        for qid, question in record["request"]["questions"].items():
            truth = target_probabilities(question, record["targets"][qid])
            if truth is None:
                continue
            keys = list(candidates(question))
            scored = scored_rows[record["index"], qid]
            row = {"id": f"{record['index']}:{qid}", "group_id": str(group), "type": question["type"],
                   "candidates": keys, "target": [truth[k] for k in keys], "status": scored["status"],
                   "gold": target_label(question, record["targets"][qid]),
                   **{k: record["meta"].get(k, "unknown") for k in ("domain", "modality", "language")}}
            if scored["status"] == "ok":
                row["probabilities"] = [scored["probabilities"][k] for k in keys]
            rows.append(row)
    cache = {"schema_version": 1, "role": role, "identity": runtime_identity(runtime, protocol),
             "group_namespace": group_namespace,
             "dataset": {"benchmark": manifest["signature"]["benchmark"],
                         "sha256": dataset.manifest["sha256"], "run_signature_sha256": digest(manifest["signature"]),
                         "predictions_sha256": prediction_hash, "manifest_sha256": manifest_hash},
             "rows": rows}
    validate_cache(cache, role)
    write_json(output, cache)
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("calibration", "policy", "test", "unassigned"), required=True)
    parser.add_argument("--protocol", required=True, help="JSON deployment recipe")
    parser.add_argument("--group-namespace", required=True)
    parser.add_argument("--root")
    parser.add_argument("--group-by", choices=("meta", "normalized-state"), default="meta")
    args = parser.parse_args(argv)
    export_run(args.run_dir, args.output, args.role, json.loads(Path(args.protocol).read_text()),
               args.group_namespace, args.root, args.group_by)


if __name__ == "__main__":
    main()
