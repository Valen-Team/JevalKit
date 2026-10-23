import copy
import json
from pathlib import Path

import pytest

from jevalkit.datasets import BENCHMARKS, DecisionDataset
from jevalkit.datasets._shared import commit_dataset
from jevalkit.io import read_tsv, write_tsv
from jevalkit.leaderboard import build_leaderboard
from jevalkit.metrics import question_metrics, summarize
from jevalkit.models.uniform import UniformModel
from jevalkit.models.openai_decisions import OpenAIDecisions
from jevalkit.runner import read_journal, run, selected_records
from jevalkit.schema import answer_probabilities, distribution, inference_request


@pytest.fixture(autouse=True)
def register_decision_fixture(monkeypatch):
    monkeypatch.setitem(BENCHMARKS, "DecisionFixture", {"family": "fixture", "modality": "text"})


def record(index="one"):
    return {
        "index": index,
        "request": {
            "state": "猫\tcat\nstate",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "Alpha", "b": "Beta"},
                },
                "condition": {"type": "noul", "instructions": "Is it true?"},
                "severity": {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]},
            },
        },
        "targets": {
            "route": {"label": "a"},
            "condition": {"label": True},
            "severity": {"probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}},
        },
        "meta": {"gold_only_sentinel": "MUST_NOT_REACH_INFERENCE"},
        "assets": [],
    }


def test_tsv_roundtrip_nested_unicode_and_labels(tmp_path):
    source = record()
    write_tsv(tmp_path / "test.tsv", [source])
    assert list(read_tsv(tmp_path / "test.tsv")) == [source]
    req = inference_request(source)
    assert set(req) == {"state", "questions"}
    assert "MUST_NOT_REACH_INFERENCE" not in json.dumps(req)
    req["questions"].clear()
    assert source == record()


def test_null_optional_noul_criteria_roundtrip_and_inference(tmp_path):
    source = record()
    source["request"]["questions"]["condition"]["criteria"] = None
    write_tsv(tmp_path / "null-noul.tsv", [source])
    restored = next(read_tsv(tmp_path / "null-noul.tsv"))
    response = UniformModel().predict(inference_request(restored))
    probabilities = answer_probabilities(
        restored["request"]["questions"]["condition"], response["answers"]["condition"]
    )
    assert restored == source and probabilities == {"false": 0.5, "true": 0.5}


@pytest.mark.parametrize(
    "state",
    [
        {
            "ticket_id": "42",
            "messages": [{"timestamp": "today", "sender": "patient", "text": "Need a visit"}],
        },
        {"messages": [{"timestamp": "today", "sender": "patient", "text": "Need a visit"}]},
        {"ticket_id": "42", "messages": [{"role": "user", "content": "Need a visit"}]},
    ],
)
def test_business_messages_remain_complete_json_evidence(state):
    from jevalkit.models._http import api_state
    from jevalkit.models.media import state_parts

    source = record()
    source["request"]["state"] = state
    before = copy.deepcopy(source)
    assert json.loads(inference_request(source)["state"]) == state
    text, images, videos = state_parts(state)
    assert json.loads(text) == state and not images and not videos
    assert json.loads(api_state(state, None)) == state
    assert json.loads(api_state(state, None, openai=True)) == state
    assert source == before


def test_real_chat_envelope_keeps_media_and_is_not_serialized(tmp_path):
    from jevalkit.models.media import state_parts

    (tmp_path / "evidence.png").write_bytes(b"image fixture")
    source = record()
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {"type": "image_url", "image_url": {"url": "evidence.png"}},
                ],
            }
        ]
    }
    source["request"]["state"] = state
    assert inference_request(source)["state"] == state
    assert state_parts(state, tmp_path) == ("Inspect this", [str(tmp_path / "evidence.png")], [])


def test_vjev_native_renderer_keeps_rubrics_and_structured_descriptions():
    from types import SimpleNamespace
    from jevalkit.models.vjev import VjevModel

    source = record()["request"]
    source["questions"]["route"]["criteria"] = {"a": None, "b": {"nested": ["α", 2]}}
    source["questions"]["condition"]["criteria"] = {
        "true": "TRUE_RUBRIC_SENTINEL",
        "false": "FALSE_RUBRIC_SENTINEL",
    }
    before = copy.deepcopy(source)
    seen = []

    def ask(state, questions, **kwargs):
        seen.append(questions)
        return UniformModel().predict(source)["answers"]

    model = object.__new__(VjevModel)
    model.engine = SimpleNamespace(ask=ask)
    model.predict(source)
    sent = seen[0]
    assert list(sent["route"]["criteria"]) == ["a", "b"]
    assert sent["route"]["criteria"]["a"] == "a"
    assert json.loads(sent["route"]["criteria"]["b"]) == {"nested": ["α", 2]}
    assert "TRUE_RUBRIC_SENTINEL" in sent["condition"]["instructions"]
    assert "FALSE_RUBRIC_SENTINEL" in sent["condition"]["instructions"]
    assert source == before


def test_strict_native_schema_gets_text_without_losing_noul_criteria():
    from jevalkit.models._utils import render_criteria

    source = record()["request"]["questions"]
    source["route"]["criteria"] = {"a": None, "b": {"nested": [True, 2]}}
    source["condition"]["criteria"] = {"true": "Match policy", "false": {"exception": "Present"}}
    result = render_criteria(source)
    assert result["route"]["criteria"]["a"] == "a"
    assert json.loads(result["route"]["criteria"]["b"]) == {"nested": [True, 2]}
    assert json.loads(result["condition"]["criteria"]["false"]) == {"exception": "Present"}
    assert source["route"]["criteria"]["a"] is None


@pytest.mark.parametrize("criteria", [None, {}])
def test_strict_sdk_uses_its_noul_defaults_for_absent_rubrics(criteria):
    from jevalkit.models._utils import render_criteria

    source = {"q": {"type": "noul", "instructions": "Is it true?", "criteria": criteria}}
    before = copy.deepcopy(source)
    assert render_criteria(source) == {"q": {"type": "noul", "instructions": "Is it true?"}}
    assert source == before


@pytest.mark.parametrize(
    "p",
    [
        {"a": 0.5},
        {"a": float("nan"), "b": 0},
        {"a": True, "b": 0},
        {"a": 0.2, "b": 0.3},
        {"a": -0.1, "b": 1.1},
    ],
)
def test_reject_invalid_distributions(p):
    with pytest.raises(ValueError):
        distribution(p, ["a", "b"])


@pytest.mark.parametrize("p", [{"a": 0.32, "b": 0.05, "c": 0.55, "d": 0.07},
                              {"a": 0.51, "b": 0.5}])
def test_probability_rounding_tolerance_includes_its_decimal_boundary(p):
    before = dict(p)
    normalized = distribution(p, list(p))
    assert sum(normalized.values()) == pytest.approx(1)
    assert p == before


@pytest.mark.parametrize("p", [{"a": 0.49, "b": 0.4999999},
                              {"a": 0.51, "b": 0.5000001}])
def test_probability_rounding_outside_the_tolerance_remains_invalid(p):
    with pytest.raises(ValueError, match="sum"):
        distribution(p, list(p))


def test_soft_targets_and_failed_denominator():
    source = record()
    q = source["request"]["questions"]["severity"]
    m = question_metrics(q, source["targets"]["severity"], {"0": 0.1, "1": 0.8, "2": 0.1})
    assert not m["hard"] and m["brier"] == 0 and m["mae"] == 0 and m["rps"] == 0
    result = summarize(
        [
            {"hard": True, "status": "ok", "correct": True, "confidence": 0.8},
            {"hard": True, "status": "unsupported"},
            {"hard": True, "status": "error"},
        ]
    )
    assert result["accuracy"] == 1 / 3 and result["coverage"] == 1 / 3
    assert result["ece"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "message",
    [
        "Example has 8795 tokens, above 8192; truncation is forbidden",
        "Question branch exceeds the 8192-token limit; no input was truncated.",
        "question 'Which response': 4131 tokens is over the limit",
        "score needs 2..10 levels",
        "question 'decision': only 126 of its 151 option markers fit in max_len=512 with "
        "head_max_len=192 spent on the question; lower head_max_len, raise max_len, "
        "or use fewer options",
    ],
)
def test_capacity_rejection_is_unsupported_not_runtime_failure(message):
    from jevalkit.runner import score_predictions

    source = record()
    predictions = [
        {
            "index": source["index"],
            "status": "error",
            "exception": "ValueError",
            "error": message,
            "latency_s": 0.1,
        }
    ]
    metrics = summarize(score_predictions([source], predictions))
    assert metrics["failed"] == 0 and metrics["unsupported"] == 3 and metrics["coverage"] == 0


def test_repeated_laya_capacity_rejections_do_not_abort_other_requests(tmp_path):
    class CapacityLimited(UniformModel):
        def predict(self, request, **kwargs):
            if request["state"] == "too many candidates":
                raise ValueError(
                    "question 'decision': only 126 of its 151 option markers fit in max_len=512 with "
                    "head_max_len=192 spent on the question; lower head_max_len, raise max_len, "
                    "or use fewer options"
                )
            return super().predict(request)

    records = [record(str(i)) for i in range(7)]
    for row in records[:6]:
        row["request"]["state"] = "too many candidates"
    root = tmp_path / "data"
    commit_dataset("DecisionFixture", records, root, {}, protocol="fixture")
    result = run(
        "Uniform",
        "DecisionFixture",
        tmp_path / "results",
        root=root,
        warmup=0,
        model_instance=CapacityLimited(),
    )
    assert result["complete"] and result["attempted_requests"] == 7
    assert result["metrics"]["overall"]["failed"] == 0
    assert result["metrics"]["overall"]["unsupported"] == 18
    assert result["metrics"]["overall"]["valid"] == 3


def test_intern_capacity_preserves_all_supplied_candidates():
    from types import SimpleNamespace
    from jevalkit.models.intern_decision import InternDecision
    from jevalkit.schema import Unsupported

    model = object.__new__(InternDecision)
    model.module = SimpleNamespace(
        ANSWER_SYMBOLS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    request = {
        "state": "input",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Classify",
                "criteria": {str(i): str(i) for i in range(77)},
            }
        },
    }
    before = copy.deepcopy(request)
    with pytest.raises(Unsupported, match="at most 62"):
        model.predict(request)
    assert request == before


def test_intern_question_limit_preserves_the_complete_request():
    from jevalkit.models.intern_decision import InternDecision
    from jevalkit.schema import Unsupported

    model = object.__new__(InternDecision)
    request = {
        "state": "shared evidence",
        "questions": {
            str(i): {"type": "noul", "instructions": "Check criterion " + str(i)} for i in range(17)
        },
    }
    before = copy.deepcopy(request)
    with pytest.raises(Unsupported, match="at most 16 questions per request"):
        model.predict(request)
    assert request == before and len(request["questions"]) == 17


def test_openai_native_schema_and_refusal():
    model = OpenAIDecisions()
    source = record()
    request = source["request"]
    payload = model.build_payload(request)
    assert set(payload) == {"model", "input", "questions"}
    assert payload["questions"][1]["type"] == "predicate"
    assert payload["questions"][2]["levels"] == [
        {"label": str(i), "description": s} for i, s in enumerate(["low", "mid", "high"])
    ]
    response = {
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "probabilities": [{"value": "b", "probability": 0.9}, {"value": "a", "probability": 0.1}],
            },
            {"type": "predicate", "name": "condition", "probability": 0.7},
            {"type": "refusal", "name": "severity"},
        ]
    }
    result = model.convert_response(response, request["questions"])
    assert answer_probabilities(request["questions"]["route"], result["answers"]["route"]) == {
        "a": 0.1,
        "b": 0.9,
    }
    assert result["answers"]["severity"] == {"type": "refusal"}
    assert source == record()
    response["answers"][0]["name"] = "wrong"
    with pytest.raises(ValueError, match="name/order"):
        model.convert_response(response, request["questions"])


def test_qev_capacity_is_explicit_without_dropping_candidates():
    from jevalkit.models.qev import QEVModel
    from jevalkit.schema import Unsupported

    model = object.__new__(QEVModel)
    source = record()["request"]
    source["questions"]["route"]["criteria"] = {str(i): str(i) for i in range(77)}
    before = copy.deepcopy(source)
    with pytest.raises(Unsupported, match="at most 16"):
        model.predict(source)
    assert source == before


def test_decision1_score_capacity_is_explicit_before_forward():
    from jevalkit.models._hf import HFSystemOne
    from jevalkit.schema import Unsupported

    model = object.__new__(HFSystemOne)
    model.image_support, model.max_score_levels = False, 10
    request = record()["request"]
    request["questions"]["severity"]["criteria"] = [str(i) for i in range(11)]
    with pytest.raises(Unsupported, match="at most 10 score levels"):
        model.predict(request)
    assert len(request["questions"]["severity"]["criteria"]) == 11


@pytest.mark.parametrize("state", ["", " \n\t"])
def test_decision1_empty_state_is_an_explicit_native_input_limit(state):
    from jevalkit.config import MODELS
    from jevalkit.models._hf import HFSystemOne
    from jevalkit.schema import Unsupported

    model = object.__new__(HFSystemOne)
    model.image_support, model.max_score_levels = False, 10
    model.requires_nonempty_state = MODELS["Decision1-Kai"].options["requires_nonempty_state"]
    request = record()["request"]
    request["state"] = state
    before = copy.deepcopy(request)
    with pytest.raises(Unsupported, match="requires a nonempty state"):
        model.predict(request)
    assert request == before


def test_model_fingerprint_covers_auxiliary_heads_and_configuration(tmp_path):
    from jevalkit.models.provenance import inference_artifacts

    for name in ("head.pt", "adapter.safetensors", "calibration.json", "tokenizer.model", "inference.py"):
        (tmp_path / name).write_text("original")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "temporary.json").write_text("download state")
    before = inference_artifacts(tmp_path)
    assert set(before) == {
        "head.pt",
        "adapter.safetensors",
        "calibration.json",
        "tokenizer.model",
        "inference.py",
    }
    (tmp_path / "head.pt").write_text("changed")
    assert inference_artifacts(tmp_path)["head.pt"] != before["head.pt"]


def test_clef_rejects_long_evidence_before_native_forward():
    from types import SimpleNamespace
    from jevalkit.models.clef import ClefModel
    from jevalkit.schema import Unsupported

    model = object.__new__(ClefModel)
    seen = []

    def encode(tokenizer, payload, **kwargs):
        seen.append((payload, kwargs))
        return SimpleNamespace(input_ids=tuple(range(17)))

    model.info, model.max_length = {"model": "fixture"}, 16
    model.module = SimpleNamespace(encode_record=encode)
    model.processor = SimpleNamespace(tokenizer=None)
    with pytest.raises(Unsupported, match="truncation is forbidden"):
        model.predict(record()["request"])
    assert seen[0][0]["state"] == record()["request"]["state"]
    assert seen[0][1]["max_length"] > 16


def test_cached_base_fingerprint_identifies_the_loaded_snapshot(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from jevalkit.models.provenance import cached_base_artifacts

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("fixture")
    seen = []

    def locate(model, filename, **kwargs):
        seen.append((model, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=locate))
    info = cached_base_artifacts("Author/Base", "pinned", "/cache")
    assert info["path"] == str(tmp_path)
    assert set(info["artifacts_sha256"]) == {"config.json", "model.safetensors"}
    assert seen[0] == ("Author/Base", "config.json", {"revision": "pinned", "cache_dir": "/cache"})


def test_omni_native_abstention_is_preserved_before_conditioning():
    from jevalkit.models.omnijev import omni_answer

    q = record()["request"]["questions"]["route"]
    answer = omni_answer(q, {"probabilities": {"a": 0.2, "b": 0.3}, "abstain": 0.5})
    assert answer["probabilities"] == pytest.approx({"a": 0.4, "b": 0.6})
    assert omni_answer(q, {"probabilities": {}, "abstain": 1})["type"] == "refusal"
    with pytest.raises(ValueError):
        omni_answer(q, {"probabilities": {"a": 0.2, "b": 0.3}, "abstain": 0.2})


def test_journal_recovery_and_corruption(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"index":"one"}\n{"ind')
    assert read_journal(path, {"one"}) == [{"index": "one"}]
    assert path.read_bytes() == b'{"index":"one"}\n'
    path.write_bytes(b'{"index":"one"}\nbroken\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_journal(path, {"one"})
    path.write_bytes(b'{"index":"one"}\n{"index":"one"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_journal(path, {"one"})


def test_inference_resume_and_leaderboard_evidence(tmp_path):
    root, out = tmp_path / "data", tmp_path / "out"
    commit_dataset("DecisionFixture", [record()], root, {"fixture": True}, protocol="test")
    result = run("Uniform", "DecisionFixture", out, root=root, warmup=0)
    assert result["complete"] and not result["smoke"]
    assert result["metrics"]["overall"]["questions"] == 3
    assert result["metrics"]["overall"]["hard_questions"] == 2
    from jevalkit.runner import score_saved_run

    assert score_saved_run(out, root=root)["metrics"] == result["metrics"]
    run("Uniform", "DecisionFixture", out, root=root, warmup=0, resume=True)
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="settings changed"):
        run("Uniform", "DecisionFixture", out, root=root, warmup=0, limit=1, resume=True)
    board = build_leaderboard([out], tmp_path / "board")
    assert len(board) == 1 and board[0]["eligible"]
    retired = tmp_path / "retired"
    retired.mkdir()
    retired_report = dict(result, benchmark="AGNews")
    (retired / "metrics.json").write_text(json.dumps(retired_report))
    board = build_leaderboard([out, retired], tmp_path / "board")
    assert len(board) == 1 and board[0]["benchmark"] == "DecisionFixture"
    assert build_leaderboard([retired], tmp_path / "board") == []
    assert (tmp_path / "board" / "leaderboard.tsv").read_text() == ""
    with (out / "predictions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="evidence changed"):
        build_leaderboard([out], tmp_path / "board")


def test_leaderboard_overview_uses_latest_complete_run_instead_of_best(tmp_path):
    from jevalkit.schema import candidates, native_answer

    class FixedRuntime(UniformModel):
        def __init__(self, correct):
            self.correct = correct

        def predict(self, request, **kwargs):
            keys = {
                "route": "a" if self.correct else "b",
                "condition": "true" if self.correct else "false",
                "severity": "0",
            }
            return {
                "answers": {
                    qid: native_answer(q, {key: float(key == keys[qid]) for key in candidates(q)})
                    for qid, q in request["questions"].items()
                }
            }

    root, output = tmp_path / "data", tmp_path / "results"
    commit_dataset("DecisionFixture", [record()], root, {}, protocol="fixture")
    run(
        "Uniform", "DecisionFixture", output / "older", root=root, warmup=0, model_instance=FixedRuntime(True)
    )
    run(
        "Uniform",
        "DecisionFixture",
        output / "newer",
        root=root,
        warmup=0,
        model_instance=FixedRuntime(False),
    )
    run(
        "Uniform",
        "DecisionFixture",
        output / "smoke",
        root=root,
        warmup=0,
        limit=1,
        model_instance=FixedRuntime(True),
    )
    build_leaderboard([output], tmp_path / "board")
    data = json.loads((tmp_path / "board" / "leaderboard.json").read_text())
    assert len(data["results"]) == 3 and len(data["primary_results"]) == 1
    assert data["primary_results"][0]["accuracy"] == 0
    assert data["primary_results"][0]["evidence"] == str(output / "newer" / "metrics.json")
    markdown = (tmp_path / "board" / "README.md").read_text()
    assert "| Uniform | 0.00% |" in markdown and "[run](<" in markdown


def test_known_input_contract_failure_stays_excluded_after_rescoring(tmp_path):
    from jevalkit.io import atomic_json
    from jevalkit.runner import score_saved_run

    root, output = tmp_path / "data", tmp_path / "output"
    commit_dataset("DecisionFixture", [record()], root, {}, protocol="fixture")
    report = run("Uniform", "DecisionFixture", output, root=root, warmup=0)
    report["validation"] = {"inference_input_conformant": False, "reason": "Fixture omitted supplied content"}
    atomic_json(output / "metrics.json", report)
    rescored = score_saved_run(output, root)
    assert not rescored["validation"]["inference_input_conformant"]
    rows = build_leaderboard([output], tmp_path / "board")
    assert not rows[0]["eligible"]
    assert json.loads((tmp_path / "board/leaderboard.json").read_text())["primary_results"] == []


def test_record_sharding_is_disjoint_and_complete(tmp_path):
    records = [record(str(i)) for i in range(40)]
    commit_dataset("DecisionFixture", records, tmp_path, {"fixture": True}, protocol="test")
    dataset = DecisionDataset("DecisionFixture", tmp_path)
    shards = [{r["index"] for r in selected_records(dataset, rank=i, world_size=4)} for i in range(4)]
    assert len(set.union(*shards)) == 40
    assert sum(map(len, shards)) == 40


def test_dataset_tamper_rejected(tmp_path):
    commit_dataset("DecisionFixture", [record()], tmp_path, {}, protocol="test")
    path = tmp_path / "datasets" / "DecisionFixture.tsv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        DecisionDataset("DecisionFixture", tmp_path)


@pytest.mark.parametrize("tamper", [False, True])
def test_downloaded_tsv_runs_without_a_local_manifest(tmp_path, monkeypatch, tamper):
    import jevalkit.datasets as datasets

    root = tmp_path / "data"
    manifest = commit_dataset("DecisionFixture", [record()], root, {}, protocol="published-fixture")
    local_manifest = root / "datasets/DecisionFixture.manifest.json"
    local_manifest.unlink()
    monkeypatch.setattr(datasets, "published_manifest", lambda name: manifest)
    if tamper:
        path = root / "datasets/DecisionFixture.tsv"
        path.write_text(path.read_text() + "\n")
        with pytest.raises(ValueError, match="checksum"):
            DecisionDataset("DecisionFixture", root)
    else:
        report = run("Uniform", "DecisionFixture", tmp_path / "result", root=root, warmup=0)
        assert report["complete"] and report["metrics"]["overall"]["failed"] == 0
        assert report["dataset_sha256"] == manifest["sha256"]
        assert not local_manifest.exists()


def test_published_catalog_covers_registered_views_and_official_scoring_inputs():
    from jevalkit.datasets import published_manifest

    catalog = {"datasets": [published_manifest(name) for name in BENCHMARKS if name != "DecisionFixture"]}
    assert {m["name"] for m in catalog["datasets"]} == set(BENCHMARKS) - {"DecisionFixture"}
    for manifest in catalog["datasets"]:
        assert manifest["schema_version"] == 1 and manifest["records"] > 0
        assert len(manifest["sha256"]) == 64 and manifest["protocol"]
    assert "/mnt/" not in json.dumps(catalog)
    assert published_manifest("DecisionIndex")["source"]["edition"] == "0.3"
    assert published_manifest("DecisionIndex")["suite_dir"] == "../raw/DecisionIndex-0.3"
    assert published_manifest("VisualDecisionBench_Image")["media_root"] == "../media/VisualDecisionBench"


def test_runner_stops_after_five_consecutive_errors_even_after_success(tmp_path):
    class FailingRuntime(UniformModel):
        calls, closed = 0, False

        def predict(self, request, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("fixture inference failure")
            return super().predict(request)

        def close(self):
            self.closed = True

    commit_dataset(
        "DecisionFixture", [record(str(i)) for i in range(8)], tmp_path / "data", {}, protocol="fixture"
    )
    model = FailingRuntime()
    with pytest.raises(RuntimeError, match="Five consecutive"):
        run(
            "Uniform",
            "DecisionFixture",
            tmp_path / "output",
            root=tmp_path / "data",
            warmup=0,
            model_instance=model,
        )
    assert model.calls == 6 and model.closed
    journal = [
        json.loads(line) for line in (tmp_path / "output" / "predictions.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in journal] == ["ok"] + ["error"] * 5


def test_registered_models_have_families_and_importable_adapters():
    import importlib
    from jevalkit.config import MODELS

    assert MODELS and all(spec.family and spec.source_url for spec in MODELS.values())
    for spec in MODELS.values():
        module, name = spec.adapter.split(":")
        assert getattr(importlib.import_module(module), name)


def test_shard_merge_requires_exact_complete_partition(tmp_path):
    from jevalkit.runner import merge_shards

    root = tmp_path / "data"
    commit_dataset("DecisionFixture", [record(str(i)) for i in range(8)], root, {}, protocol="test")
    folders = [tmp_path / ("shard" + str(i)) for i in range(2)]
    for rank in range(2):
        run("Uniform", "DecisionFixture", folders[rank], root=root, rank=rank, world_size=2, warmup=0)
    with pytest.raises(ValueError, match="Missing shards"):
        merge_shards("Uniform", "DecisionFixture", folders[:1], tmp_path / "incomplete", root=root)
    report = merge_shards("Uniform", "DecisionFixture", folders, tmp_path / "merged", root=root)
    assert report["complete"] and report["metrics"]["overall"]["questions"] == 24
    assert len(build_leaderboard([tmp_path / "merged"], tmp_path / "board")) == 1


def test_upstream_import_checks_actual_prediction_input_ids(tmp_path, monkeypatch):
    from jevalkit.import_results import import_decision_index
    from jevalkit.io import atomic_json, sha256

    model_path = tmp_path / "Valen-0.8B"
    model_path.mkdir()
    atomic_json(model_path / "config.json", {})
    env_path = tmp_path / "environment.json"
    atomic_json(
        env_path,
        {
            "model_source": {
                "model_path": str(model_path),
                "config_sha256": sha256(model_path / "config.json"),
            },
            "hardware": "fixture",
        },
    )
    source = record()
    source["meta"]["_evaluation"] = {"payload_sha256": "frozen"}
    commit_dataset("DecisionIndex", [source], tmp_path / "data", {}, protocol="fixture")
    response = UniformModel().predict(source["request"])
    results = tmp_path / "upstream.jsonl"
    results.write_text(
        json.dumps(
            {
                "run_id": "one",
                "payload_sha256": "frozen",
                "status": "ok",
                "response": response,
                "model_request_wall_ms": 10,
            }
        )
        + "\n"
    )
    monkeypatch.setattr("jevalkit.import_results.official_report", lambda *args: {"decision_index": 12})
    report = import_decision_index(
        "Valen-0.8B", results, [env_path], tmp_path / "imported", root=tmp_path / "data"
    )
    assert report["complete"] and report["runtime"]["origin"] == "imported-upstream-measurement"
    bad = results.read_text().replace('"frozen"', '"changed"')
    results.write_text(bad)
    with pytest.raises(ValueError, match="input identity mismatch"):
        import_decision_index("Valen-0.8B", results, [env_path], tmp_path / "bad", root=tmp_path / "data")
