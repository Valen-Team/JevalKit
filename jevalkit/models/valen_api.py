"""Valen's hosted System One API: the free endpoint or a keyed deployment."""

import json

from .jev import JevModel

FREE_ENDPOINT = "https://api.valen-team.workers.dev/v1/systemone"
# Fixed server limit messages; anything else stays an opaque HTTP error.
LIMIT_MESSAGES = ("At most ", "`state` renders to ", "each criterion must be at most ", "instructions must be at most ")


class ValenAPIModel(JevModel):
    """Jev's wire format, sending at most max_questions questions per request over the same state."""

    def __init__(self, *, model="valen-2b", endpoint=FREE_ENDPOINT, token_env="VALEN_API_KEY", long_context=True,
                 max_questions=8, expected_response_model=None, **options):
        size = model.rsplit("-", 1)[-1].upper()
        super().__init__(model=model, endpoint=endpoint, token_env=token_env, long_context=long_context,
                         expected_response_model=expected_response_model or "Valen-" + size, **options)
        if type(max_questions) is not int or max_questions < 1:
            raise ValueError("max_questions must be a positive integer")
        self.max_questions = max_questions

    def unsupported_http_reason(self, error):
        if error.code != 400:
            return None
        try:
            detail = str(json.load(error).get("detail", ""))
        except (ValueError, AttributeError):
            return None
        return detail[:300] if any(message in detail for message in LIMIT_MESSAGES) else None

    def predict(self, request, *, media_root=None, assets=None):
        items = list(request["questions"].items())
        responses = [
            super(ValenAPIModel, self).predict(
                {"state": request["state"], "questions": dict(items[i:i + self.max_questions])},
                media_root=media_root,
            )
            for i in range(0, len(items), self.max_questions)
        ]
        merged = {**responses[0], "answers": {}, "requests": len(responses)}
        for response in responses:
            merged["answers"].update(response["answers"])
        usage = [response.get("usage") or {} for response in responses]
        merged["usage"] = {
            key: sum(u[key] for u in usage) if all(u.get(key) is not None for u in usage) else None
            for key in ("input_tokens", "output_tokens")
        }
        return merged

    def metadata(self):
        return {**super().metadata(), "max_questions": self.max_questions, "hardware": "Valen hosted API"}
