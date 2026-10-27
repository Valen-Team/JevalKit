import contextlib
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jevalkit.io import object_hash
from jevalkit.models.valen import ValenModel
from jevalkit.reliability_export import export_run


def fixture(tmp_path):
    records = [{"index": str(i), "request": {"state": f"ticket {i}", "questions": {
        "intent": {"type": "choice", "instructions": "route", "criteria": {"a": "A", "b": "B"}}}},
        "targets": {"intent": "a"}, "meta": {"group_id": f"customer-{i}", "modality": "text"}}
        for i in range(2)]
    class Dataset:
        manifest = {"sha256": "dataset-sha"}
        def __iter__(self): return iter(records)
    manifest = {"signature": {"model": "Valen-0.8B", "benchmark": "fixture", "records": 2,
                               "dataset_sha256": "dataset-sha", "limit": None, "rank": 0, "world_size": 1,
                               "selected_ids_sha256": object_hash(["0", "1"])},
                "runtime": {"model": "Valen-Team/Valen-0.8B", "revision": "pinned", "execution": "shared_state",
                            "attention": "sdpa", "device": "cpu", "dtype": "float64", "temperature": 1.0,
                            "artifacts_sha256": {"weights": "a" * 64}}}
    predictions = [{"index": "0", "status": "ok", "latency_s": .1,
                    "response": {"answers": {"intent": {"type": "choice", "choice": "a", "probabilities": {"a": .9, "b": .1}}}}},
                   {"index": "1", "status": "error", "latency_s": .1}]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in predictions))
    return Dataset(), manifest, records


def test_export_preserves_failed_question_and_explicit_source_groups(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    with patch("jevalkit.reliability_export.DecisionDataset", return_value=dataset):
        cache = export_run(tmp_path, tmp_path / "cache.json", "test", {"recipe": "v1"}, "customers")
    assert [r["status"] for r in cache["rows"]] == ["ok", "error"]
    assert cache["rows"][0]["target"] == [1, 0]
    assert cache["identity"]["backend"].endswith(":cpu")


def test_export_rejects_altered_selected_ids_before_writing(tmp_path):
    dataset, manifest, _ = fixture(tmp_path)
    manifest["signature"]["selected_ids_sha256"] = "changed"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with patch("jevalkit.reliability_export.DecisionDataset", return_value=dataset), pytest.raises(ValueError, match="Incomplete"):
        export_run(tmp_path, tmp_path / "cache.json", "test", {}, "customers")
    assert not (tmp_path / "cache.json").exists()


def test_export_requires_groups_or_explicit_normalized_text_grouping(tmp_path):
    dataset, _, records = fixture(tmp_path)
    for record in records: record["meta"].pop("group_id")
    with patch("jevalkit.reliability_export.DecisionDataset", return_value=dataset):
        with pytest.raises(ValueError, match="group_id"):
            export_run(tmp_path, tmp_path / "cache.json", "test", {}, "customers")
        result = export_run(tmp_path, tmp_path / "cache.json", "unassigned", {}, "utterances", group_by="normalized-state")
    assert all(len(r["group_id"]) == 64 for r in result["rows"])


def test_adapter_applies_temperature_and_refuses_body_override_even_at_one():
    calls = []
    adapter = ValenModel.__new__(ValenModel)
    adapter.info = {"calibration_sha256": "bound"}
    adapter.temperature = 1.0
    adapter.execution = "shared_state"
    adapter.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    adapter.model = SimpleNamespace(predict=lambda request, **kwargs: calls.append(kwargs) or {})
    request = {"state": "hello", "questions": {"q": {"type": "choice", "instructions": "route", "criteria": {"a": "A", "b": "B"}}}}
    with pytest.raises(ValueError, match="override"):
        adapter.predict(dict(request, temperature=2))
    adapter.temperature = {"choice": 1.5}
    adapter.predict(request)
    assert calls[0]["temperature"] == {"choice": 1.5}
    assert "temperature" not in request


def test_export_rejects_prediction_tampering_against_original_metrics(tmp_path):
    from jevalkit.io import sha256
    dataset, _, _ = fixture(tmp_path)
    (tmp_path / "metrics.json").write_text(json.dumps({
        "manifest_sha256": sha256(tmp_path / "manifest.json"), "predictions_sha256": "changed"}))
    with patch("jevalkit.reliability_export.DecisionDataset", return_value=dataset), pytest.raises(ValueError, match="integrity"):
        export_run(tmp_path, tmp_path / "cache.json", "test", {}, "customers")
