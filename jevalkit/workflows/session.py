"""One case's model calls, with per-call latency, usage and estimated cost."""

import time

from ..io import object_hash
from ..pricing import cost_usd, usage_tokens


class Session:
    def __init__(self, model, rates=None, media_root=None):
        self.model, self.rates, self.media_root = model, rates, media_root
        self.calls = []

    def ask(self, state, questions):
        """Native answers keyed by question ID; a failed call raises and is recorded as failed."""
        call = {"state_sha256": object_hash(state), "questions": questions, "status": "ok"}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            response = self.model.predict({"state": state, "questions": questions}, media_root=self.media_root)
            if set(response.get("answers", {})) != set(questions):
                raise ValueError("Response question IDs differ from the requested question IDs")
        except Exception as exc:
            tokens = usage_tokens({"usage": getattr(exc, "usage", None)})
            call.update(status="error", exception=type(exc).__name__, error=str(exc), input_tokens=tokens[0],
                        output_tokens=tokens[1], cost_usd=cost_usd(self.rates, *tokens),
                        request_id=getattr(exc, "request_id", None))
            raise
        finally:
            call["latency_s"] = time.perf_counter() - started
        tokens = usage_tokens(response)
        call.update(answers=response["answers"], input_tokens=tokens[0], output_tokens=tokens[1],
                    cost_usd=cost_usd(self.rates, *tokens), request_id=response.get("request_id"))
        return response

    def totals(self):
        """Summed usage and cost; a total is unknown if any call lacks it."""
        out = {"call_count": len(self.calls), "summed_call_time_s": sum(c["latency_s"] for c in self.calls)}
        for name in ("input_tokens", "output_tokens", "cost_usd"):
            values = [c.get(name) for c in self.calls]
            out[name] = sum(values) if None not in values else None
        return out
