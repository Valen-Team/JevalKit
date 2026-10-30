"""Workflow runs, action agreement, usage and cost, the hosted Valen adapter and slice analysis."""

import io
import json
from urllib.error import HTTPError

import pytest

from jevalkit.analyze import analyze
from jevalkit.config import build_model
from jevalkit.io import atomic_json, jsonl
from jevalkit.metrics import grouped_report, score_predictions, slices
from jevalkit.models.base import DecisionModel
from jevalkit.pricing import cost_usd, rates_for
from jevalkit.schema import native_answer
from jevalkit.workflows import action, load
from jevalkit.workflows.runner import agreement, run_workflow, score_run


class Scripted(DecisionModel):
    """Answers every delivery-support message as a late order with no flags."""

    def __init__(self, provider="openai", model="gpt-4o-mini"):
        self.provider, self.model, self.requests = provider, model, []

    def predict(self, request, *, media_root=None, assets=None):
        self.requests.append(request)
        answers = {}
        for qid, question in request["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                answers[qid] = native_answer(question, {k: 0.9 if k == "late_order" else 0.1 / (len(keys) - 1)
                                                        for k in keys})
            else:
                answers[qid] = native_answer(question, {"true": 0.0, "false": 1.0})
        return {"answers": answers, "usage": {"input_tokens": 1000, "output_tokens": 100}}

    def metadata(self):
        return {"adapter": "Scripted", "provider": self.provider, "model": self.model}


def test_agreement_compares_action_sets_and_primary_action():
    route = action("route", {"queue": "late_order"})
    reference = {"actions": [route, action("safety")], "primary_action": route}
    assert agreement(reference, {"actions": [action("safety"), route, route], "primary_action": route}) == (True, True)
    assert agreement(reference, {"actions": [route], "primary_action": route}) == (False, True)
    other = action("route", {"queue": "food_quality"})
    assert agreement(reference, {"actions": [other, action("safety")], "primary_action": other}) == (False, False)
    assert agreement(reference, None) == (False, False)
    assert agreement(None, {"actions": [], "primary_action": None}) == (None, None)


def test_delivery_support_run_scores_against_human_labels_with_cost(tmp_path):
    model = Scripted()
    report = run_workflow("delivery_support", "LLM-OpenAI", tmp_path, model_instance=model, workers=3)
    workflow = load("delivery_support")
    late_without_flags = sum(
        item["expected"] == {"queue": "late_order", "safety": False, "compensation": False}
        for item in workflow.items.values()
    )
    late = sum(item["expected"]["queue"] == "late_order" for item in workflow.items.values())
    human = report["references"]["human"]
    assert report["complete"] and report["status"] == {"ok": 60, "unsupported": 0, "error": 0}
    assert human["exact_actions"] == pytest.approx(late_without_flags / 60)
    assert human["primary_action"] == pytest.approx(late / 60)
    assert report["mean_cost_usd"] == pytest.approx((1000 * 0.15 + 100 * 0.6) / 1e6)
    assert len(model.requests) == 60 and set(model.requests[0]["questions"]) >= {"queue", "safety", "compensation"}
    rows = list(jsonl(tmp_path / "cases.jsonl"))
    assert rows[0]["calls"][0]["answers"]["queue"]["choice"] == "late_order"
    assert score_run(tmp_path)["references"] == report["references"]


def test_failed_cases_stay_in_the_denominator_and_resume_checks_settings(tmp_path):
    first = load("delivery_support").cases()[0]

    class Failing(Scripted):
        def predict(self, request, **kwargs):
            if request["state"]["message"] == load("delivery_support").items[first]["message"]:
                raise RuntimeError("Decision API HTTP 503")
            return super().predict(request, **kwargs)

    report = run_workflow("delivery_support", "LLM-OpenAI", tmp_path, model_instance=Failing(), limit=5)
    assert report["cases"] == 5 and not report["complete"]
    assert report["status"] == {"ok": 4, "unsupported": 0, "error": 1}
    failed = next(r for r in jsonl(tmp_path / "cases.jsonl") if r["case_id"] == first)
    assert failed["error"] == "Decision API HTTP 503" and failed["cost_usd"] is None
    assert report["mean_cost_usd"] is None
    with pytest.raises(ValueError, match="existing run"):
        run_workflow("delivery_support", "LLM-OpenAI", tmp_path, model_instance=Scripted(), limit=5)
    with pytest.raises(ValueError, match="changed"):
        run_workflow("delivery_support", "LLM-OpenAI", tmp_path, model_instance=Scripted(), limit=4, resume=True)
    report = run_workflow("delivery_support", "LLM-OpenAI", tmp_path, model_instance=Scripted(), limit=5, resume=True)
    assert report["status"] == {"ok": 5, "unsupported": 0, "error": 0} and report["mean_cost_usd"] is not None


def test_unknown_rates_or_usage_give_unknown_cost():
    rates = rates_for({"provider": "openai", "model": "gpt-4o-mini"})
    assert cost_usd(rates, 2_000_000, None) is None
    assert cost_usd(None, 10, 10) is None
    assert rates_for({"provider": "openai", "model": "no-such-model"}) is None
    gateway = rates_for({"provider": "openai", "billing": "anthropic", "model": "claude-haiku-4-5"})
    assert gateway["key"] == "anthropic:claude-haiku-4-5"


def test_hosted_valen_splits_questions_and_reports_server_limits(monkeypatch):
    model = build_model("Valen-API-2B")
    questions = {f"q{i}": {"type": "noul", "instructions": f"Statement {i}."} for i in range(10)}
    sent = []

    def post(payload):
        sent.append(payload)
        return {"model": "Valen-2B", "usage": {"input_tokens": 50, "output_tokens": 0},
                "answers": {qid: {"type": "noul", "noul": 0.5} for qid in payload["questions"]}}

    monkeypatch.setattr(model, "post", post)
    result = model.predict({"state": "text", "questions": questions})
    assert [len(p["questions"]) for p in sent] == [8, 2]
    assert len(result["answers"]) == 10 and result["usage"] == {"input_tokens": 100, "output_tokens": 0}
    assert model.request_headers()["TypeSafe-Danger-Zone-Long-Context"] == "true"
    assert model.request_headers()["User-Agent"].startswith("jevalkit/")
    body = io.BytesIO(json.dumps({"detail": "q0: each criterion must be at most 300 characters"}).encode())
    error = HTTPError(model.endpoint, 400, "Bad Request", {}, body)
    assert model.unsupported_http_reason(error) == "q0: each criterion must be at most 300 characters"
    other = HTTPError(model.endpoint, 400, "Bad Request", {}, io.BytesIO(b'{"detail": "echoed input"}'))
    assert model.unsupported_http_reason(other) is None


def record(index, state, label):
    question = {"type": "choice", "instructions": "Which queue?", "criteria": {"a": "A", "b": "B", "c": "C"}}
    return {"index": index, "request": {"state": state, "questions": {"q": question}},
            "targets": {"q": label}, "meta": {"modality": "text"}}


def test_slices_usage_and_analysis_flag_a_weak_slice(tmp_path):
    records = [record(f"en{i}", "Short English message.", "a") for i in range(30)]
    records += [record(f"zh{i}", "我的订单还没到。" * 300, "a") for i in range(30)]
    assert slices(records[-1], records[-1]["request"]["questions"]["q"]) == \
        {"script": "cjk", "state_size": "1k-4k", "candidates": "3-4"}

    def predictions(zh_correct):
        out = []
        for r in records:
            right = r["index"].startswith("en") or int(r["index"][2:]) < zh_correct
            probabilities = {"a": 0.8, "b": 0.1, "c": 0.1} if right else {"a": 0.1, "b": 0.8, "c": 0.1}
            out.append({"index": r["index"], "status": "ok", "latency_s": 0.1,
                        "response": {"answers": {"q": native_answer(r["request"]["questions"]["q"], probabilities)},
                                     "usage": {"input_tokens": 10, "output_tokens": 2}}})
        return out

    rates = rates_for({"provider": "openai", "model": "gpt-4o-mini"})
    for name, zh_correct in (("weak", 12), ("strong", 27)):
        report = grouped_report(score_predictions(records, predictions(zh_correct), rates))
        assert report["overall"]["input_tokens_total"] == 600
        assert report["overall"]["cost_usd_total"] == pytest.approx(60 * (10 * 0.15 + 2 * 0.6) / 1e6)
        atomic_json(tmp_path / name / "metrics.json",
                    {"model": name, "benchmark": "toy", "runtime": {}, "metrics": report})
    result = analyze([tmp_path / "weak"], [tmp_path / "strong"], by=("script",))
    assert [(r["by"], r["slice"]) for r in result["flagged"]] == [("script", "cjk")]
    assert result["flagged"][0]["accuracy"] == {"weak": 0.4, "strong": 0.9}
    assert result["flagged"][0]["reference_median"] == 0.9


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def test_invoice_port_reproduces_consensus_decisions_from_consensus_answers(tmp_path):
    """Replaying the dataset's consensus answers through the vendored code must give the consensus labels."""
    pytest.importorskip("pydantic")
    parquet = pytest.importorskip("pyarrow.parquet")
    hub = pytest.importorskip("huggingface_hub")
    from jevalkit.io import object_hash
    from jevalkit.workflows.invoice_processing import REPO

    from jevalkit.paths import data_root

    try:
        path = hub.hf_hub_download(REPO, "data/questions.parquet", repo_type="dataset",
                                   cache_dir=data_root() / "cache" / "hf" / "hub", local_files_only=True)
        load("invoice_processing")
    except Exception as exc:
        pytest.skip(f"invoice_processing dataset is not in the local Hugging Face cache: {exc}")
    # The dataset serialized states with rounded floats; key answers on rounded states.
    table = {(object_hash(rounded(json.loads(row["state_json"]))), row["question_id"]):
             {p["option"]: p["probability"] for p in row["consensus"]["probabilities"]}
             for row in parquet.read_table(path).to_pylist()}

    class Replay(DecisionModel):
        def predict(self, request, *, media_root=None, assets=None):
            key = object_hash(rounded(request["state"]))
            return {"answers": {qid: native_answer(q, table[key, qid]) for qid, q in request["questions"].items()}}

    report = run_workflow("invoice_processing", "Uniform", tmp_path, model_instance=Replay(), workers=8)
    assert report["status"]["ok"] == 150
    assert report["references"]["consensus"]["exact_actions"] == 1.0
    assert report["references"]["consensus"]["primary_action"] == 1.0


def test_question_misses_lists_what_every_target_gets_wrong(tmp_path):
    from jevalkit.analyze import markdown, question_misses
    from jevalkit.datasets._shared import commit_dataset

    records = [record("easy", "Short English message.", "a"), record("hard", "Short English message.", "b")]
    commit_dataset("DecisionFixture", records, tmp_path, {"fixture": True}, protocol="test")

    def save(name, answers):
        folder = tmp_path / name
        predictions = [{"index": r["index"], "status": "ok", "latency_s": 0.1,
                        "response": {"answers": {"q": native_answer(r["request"]["questions"]["q"], answers[r["index"]])}}}
                       for r in records]
        folder.mkdir()
        (folder / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
        report = grouped_report(score_predictions(records, predictions))
        atomic_json(folder / "metrics.json", {"model": name, "benchmark": "DecisionFixture", "runtime": {}, "metrics": report})
        return folder

    right = {"easy": {"a": 0.8, "b": 0.1, "c": 0.1}, "hard": {"a": 0.1, "b": 0.8, "c": 0.1}}
    wrong_hard = {**right, "hard": {"a": 0.8, "b": 0.1, "c": 0.1}}
    targets = [save("valen-small", wrong_hard), save("valen-large", wrong_hard)]
    references = [save("ref-1", right), save("ref-2", right), save("ref-3", wrong_hard)]
    misses = question_misses(targets, references, tmp_path, min_reference_share=0.6)
    assert [(m["index"], m["expected"], m["reference_share"]) for m in misses] == [("hard", "b", pytest.approx(2 / 3))]
    assert misses[0]["predicted"] == {"valen-small": "a", "valen-large": "a"}
    assert "| hard |" in markdown(analyze(targets, references, by=("script",)), misses)
    with pytest.raises(ValueError, match="distinct model name"):
        question_misses(targets, [save("valen-small-copy", right), targets[0]], tmp_path)
