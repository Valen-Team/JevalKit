import copy
import hashlib
import json

import pytest

from jevalkit.datasets._shared import calibration_error
from jevalkit.datasets import DecisionDataset
from jevalkit.datasets._shared import acquire_files
from jevalkit.datasets.decision_bench import decisionbench_record
from jevalkit.datasets.jevalt_bench import jevalt_record
from jevalkit.datasets.procedural_typed_decisions import procedural_record
from jevalkit.datasets.typed_decisions import typed_record
from jevalkit.datasets.zh_decision_bench import zh_record
from jevalkit.datasets._shared import commit_dataset
from jevalkit.metrics import question_metrics, summarize
from jevalkit.runner import run
from jevalkit.metrics import score_predictions
from jevalkit.schema import inference_request, validate_record


def typed_source():
    return {
        "id": "typed-fixture",
        "workflow": "customer_service",
        "state": '{"message":"Need help"}',
        "questions": json.dumps(
            {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a team",
                    "criteria": {"z": "Billing", "a": "Support"},
                }
            }
        ),
        "gold": json.dumps({"route": {"label": "z", "probabilities": {"z": 0.7, "a": 0.3}}}),
        "factors": {"PRIVATE_REFERENCE_SENTINEL": True},
    }


def decision_source():
    return {
        "row_id": "decision-fixture",
        "task_id": "ops/impact/ordinal",
        "primitive": "ordinal_scoring",
        "family": "impact",
        "domain": "ops",
        "reasoning_required": False,
        "instruction": "Rate the proposed operation",
        "state_json": '{"operation":"edit"}',
        "candidates_json": json.dumps(
            [
                {"id": "high", "label": "High", "description": "Severe impact", "ordinal_value": 10},
                {"id": "low", "label": "Low", "description": None, "ordinal_value": 0},
                {"id": "medium", "label": "Medium", "description": None, "ordinal_value": 2},
            ]
        ),
        "gold_candidate_id": "low",
        "gold_probabilities": [0.0, 1.0, 0.0],
        "compact_instruction": "DO_NOT_USE_COMPACT",
        "source_json": '{"answer_key":"PRIVATE_REFERENCE_SENTINEL"}',
    }


def test_typed_soft_gold_has_independent_accuracy_and_probability_metrics():
    record = validate_record(typed_record(typed_source()))
    question = record["request"]["questions"]["route"]
    result = question_metrics(question, record["targets"]["route"], {"z": 0.6, "a": 0.4}, ["z", "a"])
    assert result["hard"] and result["correct"]
    assert result["brier"] == pytest.approx(0.02)
    assert result["hard_brier"] == pytest.approx(0.32)
    assert result["kl"] > 0
    assert record["request"]["state"] == {"message": "Need help"}
    assert "PRIVATE_REFERENCE_SENTINEL" not in json.dumps(inference_request(record))
    broken = copy.deepcopy(record)
    broken["targets"]["route"]["label"] = "missing"
    with pytest.raises(ValueError, match="not a candidate"):
        validate_record(broken)


def test_decisionbench_ordinal_reordering_restores_ids_values_and_upstream_ties():
    record = validate_record(decisionbench_record(decision_source()))
    q = record["request"]["questions"]["decision"]
    assert q["criteria"] == ["Low", "Medium", "High: Severe impact"]
    assert record["targets"]["decision"]["label"] == "0"
    assert record["targets"]["decision"]["ordinal_values"] == [0, 2, 10]
    assert record["meta"]["upstream_candidate_ids"] == {"2": "high", "0": "low", "1": "medium"}
    metric = question_metrics(
        q,
        record["targets"]["decision"],
        {"0": 0.5, "1": 0.0, "2": 0.5},
        record["meta"]["candidate_order"]["decision"],
    )
    assert metric["predicted"] == "2" and not metric["correct"]
    assert metric["mae"] == 5
    visible = json.dumps(inference_request(record))
    assert "PRIVATE_REFERENCE_SENTINEL" not in visible and "DO_NOT_USE_COMPACT" not in visible
    assert "ordinal_values" not in visible


def test_decisionbench_binary_follows_official_first_candidate_true_mapping():
    row = decision_source()
    row.update(
        primitive="binary_classification",
        candidates_json=json.dumps(
            [
                {"id": "negative", "label": "No", "description": None},
                {"id": "positive", "label": "Yes", "description": None},
            ]
        ),
        gold_candidate_id="positive",
        gold_probabilities=[0.1, 0.9],
    )
    record = validate_record(decisionbench_record(row))
    assert record["request"]["questions"]["decision"]["criteria"] == {"true": "No", "false": "Yes"}
    assert record["targets"]["decision"]["label"] == "false"
    assert record["targets"]["decision"]["probabilities"] == {"false": 0.9, "true": 0.1}
    row["gold_probabilities"] = [0.1, 0.8, 0.1]
    with pytest.raises(ValueError, match="align"):
        decisionbench_record(row)


def test_procedural_posterior_remains_soft_without_inventing_observed_outcomes():
    row = {
        "id": "posterior:test:0",
        "task": "partial_observation_calibration",
        "level": 3,
        "state": "A noisy sensor fired",
        "questions": json.dumps({"incident": {"type": "noul", "instructions": "Is the incident real?"}}),
        "answers": json.dumps({"incident": {"type": "noul", "noul": 0.3}}),
    }
    record = validate_record(procedural_record(row))
    record["targets"]["incident"]["label"] = None
    validate_record(record)
    predictions = [
        {
            "index": row["id"],
            "status": "ok",
            "latency_s": 0.1,
            "response": {"answers": {"incident": {"type": "noul", "noul": 0.3}}},
        }
    ]
    metrics = summarize(score_predictions([record], predictions))
    assert metrics["brier"] == 0 and metrics["kl"] == 0
    assert metrics["accuracy"] is None and metrics["ece"] is None and metrics["hard_questions"] == 0


def test_zh_rubric_labels_and_boolean_tie_order_are_preserved():
    row = {
        "id": "zh-fixture",
        "domain": "cs",
        "source": "authored",
        "state": "退款还没到账",
        "questions": {
            "urgency": {"type": "score", "instructions": "紧急程度", "criteria": ["一般", "紧急"]},
            "escalate": {"type": "noul", "instructions": "转人工吗"},
        },
        "gold": {"urgency": "紧急", "escalate": True},
    }
    record = validate_record(zh_record(row))
    assert record["targets"]["urgency"]["label"] == "1"
    metric = question_metrics(
        record["request"]["questions"]["escalate"],
        record["targets"]["escalate"],
        {"false": 0.5, "true": 0.5},
        record["meta"]["candidate_order"]["escalate"],
    )
    assert metric["correct"]


def test_jevalt_public_abstain_flag_adds_unknown_without_reading_gold():
    row = {
        "id": "abstain-fixture",
        "lang": "en",
        "state": "The required fact is absent",
        "license": "Apache-2.0",
        "source": "authored",
        "tags": ["group:shared-evidence"],
        "abstain": True,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": {"question": "Choose"},
                "context": "Keep this context",
                "criteria": {"a": "A", "b": "B"},
            }
        },
        "targets": {"q": {"label": "unknown", "dist": {"a": 0.1, "b": 0.1, "unknown": 0.8}}},
    }
    record = validate_record(jevalt_record(row))
    assert list(record["request"]["questions"]["q"]["criteria"]) == ["a", "b", "unknown"]
    assert "Keep this context" in record["request"]["questions"]["q"]["instructions"]
    assert record["meta"]["group_id"] == "shared-evidence"
    row["targets"]["q"]["label"] = "a"
    assert jevalt_record(row)["request"] == record["request"]
    row["abstain"] = False
    with pytest.raises(ValueError, match="Probability keys"):
        jevalt_record(row)


def test_jevalt_dictionary_rubrics_keep_insertion_order_and_probability_alignment():
    row = {
        "id": "rubric-fixture",
        "lang": "tr",
        "state": "Evidence",
        "license": "Apache-2.0",
        "source": "authored",
        "questions": {
            "s": {"type": "score", "instructions": "Rate", "criteria": {"small": "Small", "large": "Large"}}
        },
        "targets": {"s": {"label": "large", "dist": {"small": 0.2, "large": 0.8}}},
    }
    record = validate_record(jevalt_record(row))
    assert record["request"]["questions"]["s"]["criteria"] == ["Small", "Large"]
    assert record["targets"]["s"] == {"label": "1", "probabilities": {"0": 0.2, "1": 0.8}}
    assert record["meta"]["ordinal_id_mapping"]["s"] == {"small": "0", "large": "1"}


def test_source_hashes_are_checked_for_new_and_cached_files(tmp_path, monkeypatch):
    payload = b'{"example":"pinned"}\n'
    spec = {
        "revision": "0" * 40,
        "repository": "fixture/source",
        "files": {"test.jsonl": hashlib.sha256(payload).hexdigest()},
    }
    spec["name"] = "Fixture"
    source = tmp_path / "source"
    source.mkdir()
    (source / "test.jsonl").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire_files(spec, tmp_path / "root", source)
    assert not list((tmp_path / "root").rglob("*.part"))
    (source / "test.jsonl").write_bytes(payload)
    destination = acquire_files(spec, tmp_path / "root", source)
    (destination / "test.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire_files(spec, tmp_path / "root", source)


def test_calibration_protocols_distinguish_boundary_bins():
    rows = [
        {"status": "ok", "hard": True, "confidence": 0.6, "correct": True},
        {"status": "ok", "hard": True, "confidence": 0.61, "correct": False},
    ]
    assert calibration_error(rows) == pytest.approx(0.105)
    assert calibration_error(rows, right_closed=True) == pytest.approx(0.505)
    assert calibration_error(rows, equal_mass=True) == pytest.approx(0.505)


def test_prepared_soft_reference_runs_and_rescores_without_gold_inference(tmp_path):
    record = typed_record(typed_source())
    commit_dataset("TypedDecisions", [record], tmp_path / "data", {"fixture": True}, protocol="fixture-only")
    report = run("Uniform", "TypedDecisions", tmp_path / "run", root=tmp_path / "data", warmup=0)
    assert report["metrics"]["overall"]["hard_questions"] == 1
    assert report["official"]["accuracy"] == 1  # z is first, despite lexical ordering.
    assert report["official"]["kl_from_gold"] > 0
    assert report["official"]["groups"]["domain"]["customer_service"]["questions"] == 1
    from jevalkit.runner import score_saved_run

    rescored = score_saved_run(tmp_path / "run", root=tmp_path / "data")
    assert rescored["official"] == report["official"]
    assert DecisionDataset("TypedDecisions", tmp_path / "data").verify()["records"] == 1


@pytest.mark.parametrize("probabilities", [{"a": 0.5, "b": 0.49}, {"a": 0.51, "b": 0.5}])
def test_official_index_rounding_boundary_keeps_raw_probabilities(probabilities):
    import copy
    pytest.importorskip("decision_index")
    from jevalkit.datasets.decision_index import validate_official_response

    questions = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}
    response = {"model": "fixture", "answers": {"q": {"type": "choice", "choice": "a", "probabilities": probabilities}}}
    before = copy.deepcopy(response)
    assert validate_official_response(questions, response)
    assert response == before


@pytest.mark.parametrize("answer", [
    {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.48}},
    {"type": "choice", "choice": "missing", "probabilities": {"a": 0.5, "b": 0.49}},
    {"type": "choice", "choice": "a", "probabilities": {"a": 0.99}},
    {"type": "choice", "choice": "a", "probabilities": {"a": float("nan"), "b": 0.5}},
    {"type": "noul", "noul": 0.5},
])
def test_official_index_boundary_compatibility_rejects_real_invalid_answers(answer):
    pytest.importorskip("decision_index")
    from jevalkit.datasets.decision_index import validate_official_response

    questions = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}
    with pytest.raises((ValueError, KeyError, TypeError)):
        validate_official_response(questions, {"answers": {"q": answer}})


def test_official_scorer_receives_original_answers_after_boundary_validation(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace
    pytest.importorskip("decision_index")
    from jevalkit.datasets.decision_index import evaluate

    monkeypatch.setattr("decision_index.suite.io.Suite", lambda *args: object())
    captured = []
    def score(suite, path, *args):
        captured.extend(json.loads(line) for line in path.read_text().splitlines())
        return {"complete": True}
    monkeypatch.setattr("decision_index.pipeline.score_run", score)
    dataset = SimpleNamespace(path=tmp_path/"data.tsv", manifest={"suite_dir": "suite", "source": {"edition": "0.3"}})
    response = {"model": "fixture", "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.49}}}}
    before = copy.deepcopy(response)
    record = {"index": "fixture", "request": {"questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}},
              "meta": {"_evaluation": {"run_id": "fixture"}}}
    report = evaluate(dataset, [record], [{"index": "fixture", "status": "ok", "latency_s": 0.1, "response": response}], tmp_path)
    assert captured[0]["response"] == before and captured[0]["status"] == "ok"
    assert response == before and report["rounding_boundary_validation_checks"] == 1
