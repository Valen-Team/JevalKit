"""Native payload preservation and independent public-index checks."""
import copy
import json
from types import SimpleNamespace

import pytest

from jevalkit.datasets import prepare_dataset
from jevalkit.datasets.decision_index_vision import evaluate, vision_record
from jevalkit.io import read_tsv, write_tsv
from jevalkit.schema import inference_request, validate_record


def test_vision_applies_the_official_valen_budget_without_changing_caller_options():
    from jevalkit.datasets import configure_model_options
    options = {"model": "local-native-model"}
    dataset = SimpleNamespace(name="DecisionIndex_Vision")
    for name in ["Valen-0.8B", "Valen-2B", "Valen-4B"]:
        configured = configure_model_options(dataset, name, options)
        assert configured == {**options, "max_image_pixels": 1638400}
    assert options == {"model": "local-native-model"}
    with pytest.raises(ValueError, match="1638400"):
        configure_model_options(dataset, "Valen-0.8B", {"max_image_pixels": 262144})
    assert configure_model_options(SimpleNamespace(name="DecisionIndex_Text"), "Valen-0.8B", options) == options


def source_row(index="image-case", benchmark="CV-Bench"):
    return {
        "id": index, "benchmark": benchmark, "set": "fixture", "subtask": "fixture",
        "images": ["full/fixture/img/second.png", "full/fixture/img/first.png"],
        "state": {},
        "questions": {"answer": {"type": "choice", "instructions": "Compare [image 1] and [image 2]",
                                  "criteria": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}}},
        "expected": {"answer": "A"},
        "prompt": {"options_in_image": True, "readout": "The answer is ("},
        "_evaluation": {"run_id": index, "benchmark": benchmark, "set": "fixture",
                        "payload_sha256": "frozen-payload", "image_sha256": ["a" * 64, "b" * 64]},
    }


def test_vision_tsv_preserves_image_order_questions_and_official_provenance(tmp_path):
    row = source_row()
    original = copy.deepcopy(row)
    record = validate_record(vision_record(row))
    write_tsv(tmp_path / "vision.tsv", [record])
    assert list(read_tsv(tmp_path / "vision.tsv")) == [record]
    request = inference_request(record)
    assert request["questions"] == row["questions"]
    assert [p["image_url"]["url"] for p in request["state"]["messages"][0]["content"]] == row["images"]
    assert record["assets"] == [{"path": path, "sha256": digest}
                                 for path, digest in zip(row["images"], row["_evaluation"]["image_sha256"])]
    assert record["meta"]["official_prompt"] == row["prompt"]
    assert record["meta"]["_evaluation"] == row["_evaluation"]
    assert set(request) == {"state", "questions"} and "frozen-payload" not in json.dumps(request)
    assert row == original


def test_vision_refuses_missing_images_and_unhandled_source_state():
    row = source_row()
    row["_evaluation"]["image_sha256"].pop()
    with pytest.raises(ValueError, match="checksummed"):
        vision_record(row)
    row = source_row()
    row["state"] = {"information": "must not be silently dropped"}
    with pytest.raises(ValueError, match="drop source content"):
        vision_record(row)


def test_track_defaults_keep_the_legacy_edition_separate(monkeypatch, tmp_path):
    import jevalkit.datasets.decision_index as legacy
    import jevalkit.datasets.decision_index_text as text
    import jevalkit.datasets.decision_index_vision as vision

    calls = []
    for module in [legacy, text, vision]:
        monkeypatch.setattr(module, "prepare", lambda **kw: calls.append(kw))
    for name in ["DecisionIndex", "DecisionIndex_Text", "DecisionIndex_Vision"]:
        prepare_dataset(name, root=tmp_path)
    assert [kw["edition"] for kw in calls] == ["0.3", "0.4.1", "0.4.1"]


@pytest.mark.parametrize("official,eligible", [
    ({"complete": False, "counts": {"ok": 1}}, False),
    ({"complete": True, "counts": {"error": 1}}, False),
    ({"complete": True, "counts": {"ok": 1}}, True),
])
def test_official_validation_and_coverage_gate_ranked_results(tmp_path, monkeypatch, official, eligible):
    from jevalkit.datasets._shared import commit_dataset
    from jevalkit.leaderboard import build_leaderboard
    from jevalkit.runner import run

    root = tmp_path / "data"
    commit_dataset("DecisionIndex_Vision", [vision_record(source_row())], root, {"kind": "fixture"}, protocol="fixture")
    monkeypatch.setattr("jevalkit.datasets.official_report", lambda *args: official)
    output = tmp_path / "run"
    run("Uniform", "DecisionIndex_Vision", output, root=root, warmup=0)
    rows = build_leaderboard([output], tmp_path / "board")
    assert len(rows) == 1 and rows[0]["eligible"] is eligible


@pytest.mark.parametrize("unsupported_first", [False, True])
def test_official_vision_index_uses_all_benchmarks_chance_weights_and_unanswered_penalty(tmp_path, unsupported_first):
    from decision_index.vision.spec import BENCHMARKS

    rows, records, predictions = [], [], []
    for benchmark in BENCHMARKS:
        for n in range(2):
            row = source_row(benchmark["name"] + str(n), benchmark["name"])
            rows.append(row)
            records.append(vision_record(row))
            choice = "A" if n == 0 else "B"
            prediction = {"index": row["id"], "status": "ok", "latency_s": 0.01,
                          "response": {"answers": {"answer": {"type": "choice", "choice": choice,
                                      "probabilities": {k: float(k == choice) for k in row["questions"]["answer"]["criteria"]}}}}}
            if unsupported_first and benchmark["name"] == "R-Bench-M" and n == 0:
                prediction = {"index": row["id"], "status": "unsupported", "latency_s": 0.01}
            predictions.append(prediction)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "vision-rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "datasets").mkdir()
    dataset = SimpleNamespace(path=tmp_path / "datasets/DecisionIndex_Vision.tsv", manifest={"suite_dir": "../raw"})
    result = evaluate(dataset, records, predictions, tmp_path)
    # All benchmarks have accuracy 1/2 and chance 1/4: independent skill 1/3.
    # Losing the sole correct R-Bench-M answer removes its weight (0.5/9.75).
    expected = (100 / 3) * (1 - 0.5 / 9.75) if unsupported_first else 100 / 3
    assert result["public_vision_index"] == pytest.approx(expected)
    assert result["decision_index"] == result["public_vision_index"]
    assert result["complete"] and result["completed"] == 22
    assert len(result["benchmarks"]) == 11
    assert result["score_scope"] == "public_vision"
    assert isinstance(result["full_index"], str)
    if unsupported_first:
        assert result["counts"]["unsupported"] == 1
        assert result["benchmarks"]["R-Bench-M"]["answered"] == 1


@pytest.mark.parametrize("budget", [False, 0, 65535, 1.5])
def test_valen_rejects_invalid_image_budgets_before_loading_a_model(budget):
    from jevalkit.models.valen import ValenModel
    with pytest.raises(ValueError, match="max_image_pixels"):
        ValenModel(model="must-not-download", max_image_pixels=budget)
