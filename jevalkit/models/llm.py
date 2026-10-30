"""General-purpose LLMs as decision-model baselines, answering with structured JSON probabilities."""

import json
import os

from ._http import HttpDecisionModel, api_state
from ._utils import render_criteria
from ..schema import candidates, native_answer

ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}
TOKEN_ENVS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
SYSTEM = (
    "You answer typed questions about the evidence. For every question, give a probability for each of its "
    "candidates; the probabilities of one question sum to 1. A noul question has the candidates true and false. "
    "A score question has its levels as candidates, keyed by zero-based index. Answer each question on its own."
)


class LLMDecisionModel(HttpDecisionModel):
    """OpenAI-compatible Chat Completions, Anthropic Messages or Gemini generateContent with a JSON schema."""

    retryable_statuses = HttpDecisionModel.retryable_statuses | {529}

    def __init__(self, provider="openai", model="gpt-4o-mini", endpoint=None, token_env=None, max_tokens=4096,
                 billing=None, **options):
        if provider not in ENDPOINTS:
            raise ValueError("provider must be one of: " + ", ".join(ENDPOINTS))
        super().__init__(
            endpoint=endpoint or ENDPOINTS[provider].format(model=model),
            model=model,
            token_env=token_env or TOKEN_ENVS[provider],
            **options,
        )
        self.provider, self.max_tokens = provider, int(max_tokens)
        # The lab whose list price applies when an OpenAI-compatible gateway serves another lab's model.
        self.billing = billing or provider

    def request_headers(self):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.token_env)
        if token and self.provider == "anthropic":
            headers.update({"x-api-key": token, "anthropic-version": "2023-06-01"})
        elif token and self.provider == "gemini":
            headers["x-goog-api-key"] = token
        elif token:
            headers["Authorization"] = "Bearer " + token
        return headers

    @staticmethod
    def evidence_parts(state, media_root):
        """Text and (mime, base64) image parts in evidence order."""
        state = api_state(state, media_root)
        if isinstance(state, str):
            return [("text", state)]
        parts = []
        for message in state["messages"]:
            for part in message["content"]:
                if part["type"] == "text":
                    parts.append(("text", part["text"]))
                else:
                    header, data = part["image_url"]["url"].split(",", 1)
                    parts.append(("image", header.removeprefix("data:").removesuffix(";base64"), data))
        return parts

    @staticmethod
    def schema(questions):
        def strict(properties):
            return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

        return strict({
            qid: strict({key: {"type": "number"} for key in candidates(q)})
            for qid, q in questions.items()
        })

    def build_payload(self, request, media_root=None):
        questions = render_criteria(request["questions"])
        listing = {
            qid: {"type": q["type"], "instructions": q["instructions"], "candidates": candidates(q)}
            for qid, q in questions.items()
        }
        prompt = "Questions:\n" + json.dumps(listing, ensure_ascii=False, indent=2)
        parts = [*self.evidence_parts(request["state"], media_root), ("text", prompt)]
        schema = self.schema(questions)
        if self.provider == "anthropic":
            content = [
                {"type": "text", "text": p[1]} if p[0] == "text"
                else {"type": "image", "source": {"type": "base64", "media_type": p[1], "data": p[2]}}
                for p in parts
            ]
            return {
                "model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM,
                "messages": [{"role": "user", "content": content}],
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            }
        if self.provider == "gemini":
            return {
                "systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"role": "user", "parts": [
                    {"text": p[1]} if p[0] == "text" else {"inlineData": {"mimeType": p[1], "data": p[2]}}
                    for p in parts
                ]}],
                "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema},
            }
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                {"type": "text", "text": p[1]} if p[0] == "text"
                else {"type": "image_url", "image_url": {"url": f"data:{p[1]};base64,{p[2]}"}}
                for p in parts
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answers", "strict": True, "schema": schema}},
        }

    def parse_response(self, response):
        """Return the model's JSON text and token usage."""
        if self.provider == "anthropic":
            if response.get("stop_reason") == "max_tokens":
                raise ValueError("Anthropic output reached max_tokens; raise --option max_tokens")
            usage = response.get("usage", {})
            text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
            return text, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
        if self.provider == "gemini":
            usage = response.get("usageMetadata", {})
            text = "".join(part.get("text", "") for part in response["candidates"][0]["content"]["parts"])
            # Gemini bills thinking tokens as output, as OpenAI and Anthropic count them.
            output = usage.get("candidatesTokenCount")
            return text, {"input_tokens": usage.get("promptTokenCount"),
                          "output_tokens": None if output is None else output + usage.get("thoughtsTokenCount", 0)}
        usage = response.get("usage") or {}
        return response["choices"][0]["message"]["content"], {
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        }

    def predict(self, request, *, media_root=None, assets=None):
        response = self.post(self.build_payload(request, media_root))
        text, usage = self.parse_response(response)
        try:
            raw, answers = self.read_answers(request, text)
        except (KeyError, TypeError, ValueError) as exc:
            exc.usage, exc.request_id = usage, response.get("id")  # An unusable answer is still billed.
            raise
        return {"answers": answers, "usage": usage, "model": self.model, "raw_probabilities": raw,
                "request_id": response.get("id")}

    @staticmethod
    def read_answers(request, text):
        text = text.strip()
        # Gateways that drop the structured-output setting return the JSON inside a Markdown fence.
        if text.startswith("```") and text.endswith("```") and len(text) > 6:
            text = text[3:-3].strip()
            text = text[4:] if text.startswith("json") else text
        raw = json.loads(text)
        answers = {}
        for qid, q in request["questions"].items():
            values = {key: float(raw[qid][key]) for key in candidates(q)}
            total = sum(values.values())
            if total <= 0:
                raise ValueError(f"LLM returned zero probability mass for {qid}")
            answers[qid] = native_answer(q, {key: value / total for key, value in values.items()})
        return raw, answers

    def metadata(self):
        return {**super().metadata(), "provider": self.provider, "billing": self.billing, "max_tokens": self.max_tokens}
