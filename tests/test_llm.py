"""General LLM baselines: provider payloads, image transport and probability conversion."""

import json

import pytest

from jevalkit.config import build_model

REQUEST = {
    "state": "I moved. Please ship my order to 12 Elm Street instead.",
    "questions": {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"change_address": "Change the address", "cancel": "Cancel the order"}},
        "refund": {"type": "noul", "instructions": "The customer asks for a refund."},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]},
    },
}
ANSWER = {"intent": {"change_address": 0.8, "cancel": 0.4}, "refund": {"false": 0.9, "true": 0.1},
          "urgency": {"0": 0.2, "1": 0.5, "2": 0.3}}


def provider_response(provider, text):
    if provider == "anthropic":
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 120, "output_tokens": 40}}
    if provider == "gemini":
        return {"candidates": [{"content": {"parts": [{"text": text}]}}],
                "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40}}
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 120, "completion_tokens": 40}}


@pytest.mark.parametrize("name,provider", [("LLM-OpenAI", "openai"), ("LLM-Anthropic", "anthropic"),
                                           ("LLM-Gemini", "gemini")])
def test_llm_baselines_convert_json_probabilities(monkeypatch, name, provider):
    model = build_model(name)
    sent = []
    monkeypatch.setattr(model, "post", lambda payload: sent.append(payload) or provider_response(provider, json.dumps(ANSWER)))
    result = model.predict(REQUEST)
    answers = result["answers"]
    assert answers["intent"]["choice"] == "change_address"
    assert answers["intent"]["probabilities"]["change_address"] == pytest.approx(2 / 3)
    assert answers["refund"] == {"type": "noul", "noul": pytest.approx(0.1)}
    assert answers["urgency"]["score"] == pytest.approx(1.1)
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 40}
    schema = json.dumps(sent[0])
    assert '"change_address"' in schema and '"additionalProperties": false' in schema
    assert model.metadata()["provider"] == provider


@pytest.mark.parametrize("provider,image_key", [("openai", "image_url"), ("anthropic", "source"), ("gemini", "inlineData")])
def test_llm_baselines_send_images(tmp_path, provider, image_key):
    (tmp_path / "photo.png").write_bytes(b"fixture image bytes")
    state = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Inspect the product photo."},
        {"type": "image_url", "image_url": {"url": "photo.png"}},
    ]}]}
    model = build_model("LLM-" + {"openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Gemini"}[provider])
    payload = model.build_payload({**REQUEST, "state": state}, media_root=tmp_path)
    assert image_key in json.dumps(payload)
    assert "Inspect the product photo." in json.dumps(payload)


def test_llm_provider_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini")
    assert build_model("LLM-Anthropic").request_headers()["x-api-key"] == "fixture-anthropic"
    gemini = build_model("LLM-Gemini", model="gemini-test")
    assert gemini.request_headers()["x-goog-api-key"] == "fixture-gemini"
    assert gemini.endpoint.endswith("/models/gemini-test:generateContent")
