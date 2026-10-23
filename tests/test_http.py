import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from jevalkit.config import build_model
from jevalkit.models._http import HttpDecisionModel
from jevalkit.models.openai_decisions import OpenAIDecisions
from jevalkit.schema import Unsupported, answer_probabilities


@pytest.fixture
def decision_server(monkeypatch):
    """Real HTTP transport with deterministic fixture outputs, never leaderboard data."""
    calls = []
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            calls.append(
                (
                    self.path,
                    dict(self.headers),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            if self.path in {"/retry", "/jev-retry"} and len(calls) == 1:
                self.send_response(529 if self.path == "/jev-retry" else 429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            if self.path.startswith("/error/"):
                self.send_response(int(self.path.rsplit("/", 1)[-1]))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = calls[-1][2]["state"]
                self.wfile.write(body.encode() if isinstance(body, str) else json.dumps(body).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"answers": {"check": {"type": "noul", "noul": 0.75}}}
            if calls[-1][2].get("model", "").startswith("jev-"):
                response["model"] = calls[-1][2]["model"]
            if calls[-1][2].get("model", "").startswith("startlux-models/"):
                response["model"] = calls[-1][2]["model"].split("/")[-1]
            if self.path == "/v1/decisions":
                response = {
                    "model": "fixture-decisions",
                    "usage": {"input_tokens": 12},
                    "answers": [
                        {
                            "name": "route",
                            "type": "choice",
                            "probabilities": [
                                {"value": "support", "probability": 0.8},
                                {"value": "billing", "probability": 0.2},
                            ],
                        },
                        {"name": "check", "type": "predicate", "probability": 0.75},
                        {
                            "name": "severity",
                            "type": "score",
                            "probabilities": [
                                {"value": "1", "probability": 0.6},
                                {"value": "0", "probability": 0.4},
                            ],
                        },
                    ],
                }
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(state="Evidence"):
    return {"state": state, "questions": {"check": {"type": "noul", "instructions": "Is this true?"}}}


def test_transport_retries_and_reads_token_without_persisting_it(decision_server, monkeypatch):
    endpoint, calls = decision_server
    monkeypatch.setenv("JEVAL_TEST_TOKEN", "test-credential")
    model = HttpDecisionModel(endpoint + "/retry", "fixture", token_env="JEVAL_TEST_TOKEN")
    result = model.predict(request())
    assert len(calls) == 2 and calls[1][1]["Authorization"] == "Bearer test-credential"
    assert answer_probabilities(request()["questions"]["check"], result["answers"]["check"])["true"] == 0.75
    assert "test-credential" not in json.dumps(model.metadata())


@pytest.mark.parametrize("model_name", ["Jev", "Kev", "H2O-Lightning", "OpenJev",
                                       "StartLux-v1.1-4B-Instant"])
def test_registered_systemone_api_uses_named_questions(decision_server, model_name):
    endpoint, calls = decision_server
    model = build_model(model_name, endpoint=endpoint + "/v1/systemone")
    source = request()
    before = copy.deepcopy(source)
    assert model.predict(source)["answers"]["check"]["type"] == "noul"
    assert calls[0][0] == "/v1/systemone" and calls[0][2]["questions"] == source["questions"]
    assert source == before


def test_jev_preserves_json_and_uses_native_asset_headers(decision_server, monkeypatch):
    endpoint, calls = decision_server
    monkeypatch.setenv("JEVAL_TEST_JEV_TOKEN", "fixture-jev-token")
    model = build_model("Jev", endpoint=endpoint + "/v1/systemone",
                        token_env="JEVAL_TEST_JEV_TOKEN", long_context=True)
    source = request({"item": {"color": "red", "quantity": 2}})
    before = copy.deepcopy(source)
    assert model.predict(source)["model"] == "jev-1.14.0"
    _, headers, sent = calls[0]
    assert sent["state"] == source["state"] and sent["questions"] == source["questions"]
    assert headers["Authorization"] == "Bearer fixture-jev-token"
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    assert normalized_headers["typesafe-danger-zone-images"] == "true"
    assert normalized_headers["typesafe-danger-zone-long-context"] == "true"
    assert "fixture-jev-token" not in json.dumps(model.metadata())
    assert model.metadata()["media_format"] == "assets"
    assert source == before


def test_jev_assets_preserve_image_bytes_and_evidence_order(tmp_path, decision_server):
    import base64

    endpoint, calls = decision_server
    data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXZkAAAAASUVORK5CYII=")
    (tmp_path / "picture.png").write_bytes(data)
    source = request({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Before"},
        {"type": "image_url", "image_url": {"url": "picture.png"}},
        {"type": "text", "text": "Between"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
        {"type": "text", "text": "After"},
    ]}]})
    before = copy.deepcopy(source)
    model = build_model("Jev", endpoint=endpoint + "/v1/systemone")
    model.predict(source, media_root=tmp_path)
    sent = calls[0][2]
    content = sent["state"]["messages"][0]["content"]
    assert content[::2] == ["Before", "Between", "After"]
    assert content[1] != content[3] and set(sent["assets"]) == {content[1], content[3]}
    for asset in sent["assets"].values():
        assert asset["type"] == "image" and asset["format"] == "png"
        assert base64.b64decode(asset["data_base64"]) == data
    assert "typesafe-danger-zone-long-context" not in {key.lower() for key in calls[0][1]}
    assert source == before


@pytest.mark.parametrize("kind,url,options,reason", [
    ("video_url", "clip.mp4", {}, "native video"),
    ("image_url", "picture.png", {"images": False}, "images=true"),
    ("image_url", "https://example.invalid/picture.png", {}, "Materialize"),
])
def test_jev_unsupported_evidence_is_rejected_before_http(decision_server, kind, url, options, reason):
    endpoint, calls = decision_server
    model = build_model("Jev", endpoint=endpoint + "/v1/systemone", **options)
    source = request({"messages": [{"role": "user", "content": [{"type": kind, kind: {"url": url}}]}]})
    with pytest.raises(Unsupported, match=reason):
        model.predict(source)
    assert calls == []


def test_jev_rejects_another_response_model(decision_server, monkeypatch):
    endpoint, calls = decision_server
    model = build_model("Jev", endpoint=endpoint + "/v1/systemone")
    monkeypatch.setattr(model, "post", lambda payload: {"model": "jev-1.13.0", "answers": {}})
    with pytest.raises(ValueError, match="expected_response_model"):
        model.predict(request())
    assert calls == []


def test_jev_retries_native_overload_status(decision_server, monkeypatch):
    endpoint, calls = decision_server
    monkeypatch.setattr("jevalkit.models._http.time.sleep", lambda seconds: None)
    model = build_model("Jev", endpoint=endpoint + "/jev-retry", retries=1)
    assert model.predict(request())["model"] == "jev-1.14.0"
    assert len(calls) == 2


def startlux_rejection(message, location=None):
    return {"error": message, "detail": [{"loc": location or ["body", "questions", "check"],
                                           "msg": message, "type": "value_error"}]}


@pytest.mark.parametrize("body,reason", [
    (startlux_rejection("the engine refused this question: prompt has 5000 tokens, exceeding the context length of 4096"), "context limit"),
    (startlux_rejection("the engine refused this question: This model's maximum context length is 4096 tokens. However, you requested 5000 tokens."), "context limit"),
    (startlux_rejection("the engine refused this question: The decoder prompt (length 5000) is longer than the maximum model length of 4096."), "context limit"),
    (startlux_rejection("this model folder has no vision tower: it reads text only", ["body", "images"]), "image input"),
    (startlux_rejection("at most 2 images per request, not 3", ["body", "images"]), "image limit"),
])
def test_startlux_http_capacity_rejections_are_unsupported(decision_server, body, reason):
    endpoint, calls = decision_server
    model = build_model("StartLux-v1.1-4B-Instant", endpoint=endpoint + "/error/422")
    with pytest.raises(Unsupported, match=reason) as error:
        model.predict(request(body))
    assert len(calls) == 1
    assert body["error"] not in str(error.value)


@pytest.mark.parametrize("code,body", [
    (422, startlux_rejection("instructions are required for choice and score questions")),
    (422, startlux_rejection("the engine refused this question: invalid tensor shape; PRIVATE_SENTINEL")),
    (422, "PRIVATE_SENTINEL not JSON"),
    (422, ["PRIVATE_SENTINEL"]),
    (422, {"error": "PRIVATE_SENTINEL", "detail": [None]}),
    (401, startlux_rejection("the engine refused this question: prompt has 5000 tokens, exceeding context length 4096")),
])
def test_startlux_http_other_errors_stay_errors_without_response_content(decision_server, code, body):
    endpoint, _ = decision_server
    model = build_model("StartLux-v1.1-4B-Instant", endpoint=endpoint + f"/error/{code}")
    with pytest.raises(RuntimeError) as error:
        model.predict(request(body))
    assert str(error.value) == f"Decision API HTTP {code}"


def test_startlux_capacity_rejections_do_not_abort_warmup_or_full_run(tmp_path, monkeypatch, decision_server):
    from jevalkit.datasets import BENCHMARKS
    from jevalkit.datasets._shared import commit_dataset
    from jevalkit.runner import run

    endpoint, calls = decision_server
    benchmark = "StartLuxCapacityFixture"
    monkeypatch.setitem(BENCHMARKS, benchmark, {"family": "fixture", "modality": "text"})
    body = startlux_rejection("the engine refused this question: prompt has 5000 tokens, exceeding context length 4096")
    records = [{"index": str(i), "request": request(body), "targets": {"check": {"label": True}},
                "meta": {}, "assets": []} for i in range(6)]
    commit_dataset(benchmark, records, tmp_path, {"kind": "synthetic HTTP fixture"}, protocol="fixture")
    output = tmp_path / "run"
    report = run("StartLux-v1.1-4B-Instant", benchmark, output, root=tmp_path,
                 model_options={"endpoint": endpoint + "/error/422"})
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert report["complete"] and len(predictions) == 6 and len(calls) == 7
    assert all(row["status"] == "unsupported" for row in predictions)


def test_openjev_screenshot_uses_author_state_image_field(tmp_path, decision_server):
    endpoint, calls = decision_server
    (tmp_path / "picture.png").write_bytes(b"fixture image bytes")
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect"},
                    {"type": "image_url", "image_url": {"url": "picture.png"}},
                ],
            }
        ]
    }
    model = build_model("OpenJev", endpoint=endpoint + "/v1/systemone")
    model.predict(request(state), media_root=tmp_path)
    sent = calls[0][2]["state"]
    assert sent["text"] == "Inspect" and sent["image"].startswith("data:image/png;base64,")
    state["messages"][0]["content"].append(state["messages"][0]["content"][-1])
    with pytest.raises(Unsupported, match="one screenshot"):
        model.predict(request(state), media_root=tmp_path)


def test_openai_image_and_video_contract(tmp_path):
    (tmp_path / "picture.png").write_bytes(b"fixture image bytes")
    part = {"type": "image_url", "image_url": {"url": "picture.png"}}
    state = {"messages": [{"role": "user", "content": [part]}]}
    native = OpenAIDecisions().build_payload(request(state), media_root=tmp_path)
    assert native["input"][0]["content"][0]["type"] == "input_image"
    assert native["input"][0]["content"][0]["image_url"].startswith("data:image/png;base64,")
    state["messages"][0]["content"] = [{"type": "video_url", "video_url": {"url": "clip.mp4"}}]
    with pytest.raises(Unsupported, match="native video"):
        OpenAIDecisions().build_payload(request(state), media_root=tmp_path)


def test_registered_openai_decisions_transport_all_question_types(decision_server, monkeypatch):
    endpoint, calls = decision_server
    monkeypatch.setenv("JEVAL_TEST_OPENAI_TOKEN", "fixture-openai-token")
    model = build_model(
        "OpenAI-Decisions", endpoint=endpoint + "/v1/decisions", token_env="JEVAL_TEST_OPENAI_TOKEN"
    )
    source = {
        "state": "Please help fix an integration error.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Route this request",
                "criteria": {"billing": None, "support": {"team": "technical"}},
            },
            "check": {
                "type": "noul",
                "instructions": "Requires support?",
                "criteria": {"true": "Technical integration issue", "false": "Billing question"},
            },
            "severity": {"type": "score", "instructions": "Rate severity", "criteria": ["low", "high"]},
        },
    }
    before = copy.deepcopy(source)
    result = model.predict(source)
    path, headers, payload = calls[0]
    assert len(calls) == 1 and path == "/v1/decisions"
    assert headers["Authorization"] == "Bearer fixture-openai-token"
    assert [q["name"] for q in payload["questions"]] == ["route", "check", "severity"]
    assert payload["questions"][0]["choices"] == [
        {"value": "billing", "description": "billing"},
        {"value": "support", "description": '{"team":"technical"}'},
    ]
    assert "Technical integration issue" in payload["questions"][1]["instructions"]
    assert "Billing question" in payload["questions"][1]["instructions"]
    assert payload["questions"][2]["levels"] == [
        {"label": "0", "description": "low"},
        {"label": "1", "description": "high"},
    ]
    assert answer_probabilities(source["questions"]["route"], result["answers"]["route"]) == {
        "billing": 0.2,
        "support": 0.8,
    }
    assert answer_probabilities(source["questions"]["check"], result["answers"]["check"]) == {
        "false": 0.25,
        "true": 0.75,
    }
    assert answer_probabilities(source["questions"]["severity"], result["answers"]["severity"]) == {
        "0": 0.4,
        "1": 0.6,
    }
    assert result["usage"] == {"input_tokens": 12}
    assert source == before and "fixture-openai-token" not in json.dumps(model.metadata())
