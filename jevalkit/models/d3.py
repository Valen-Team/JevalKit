"""Native vLLM-SR d3 adapter: the checkpoint's own Decision Index engine."""

import json

from .base import NativeRuntime
from ._utils import engine_answers, import_author_module
from .media import state_parts
from ..schema import Unsupported, is_capacity_error


class D3Model(NativeRuntime):
    """vLLM-SR d3 through ``d3_engine.D3Engine`` shipped with the checkpoint.

    Author defaults apply: a 32,768-token question cap, 1,638,400-pixel images and no permutation averaging.
    Score questions take the engine's supported level range. Video is not passed to the engine.
    """

    modalities = frozenset({"text", "image"})

    def __init__(self, model, **options):
        super().__init__(model, **options)
        engine = import_author_module(self.folder, "d3_engine")
        self.levels = import_author_module(self.folder, "d3_runtime").SCORE_LEVELS
        device = "cuda:0" if self.device == "cuda" else self.device
        self.engine = engine.D3Engine(model=str(self.folder), device=device, **self.options)
        self.info.update(
            provenance=json.loads(json.dumps(self.engine.provenance, default=str)),
            max_length=self.engine.decision.max_length,
            score_levels=list(self.levels),
        )

    def predict(self, request, *, media_root=None, assets=None):
        state, images, videos = state_parts(request["state"], media_root)
        if videos:
            raise Unsupported("d3 video input is not enabled in this adapter; all evidence was retained")
        low, high = self.levels
        questions = {}
        for qid, q in request["questions"].items():
            if q["type"] == "score" and not low <= len(q["criteria"]) <= high:
                raise Unsupported(f"d3 supports {low} to {high} score levels; all levels were retained")
            questions[qid] = {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
        try:
            response, _ = self.engine(state=state, questions=questions, images=images or None)
        except Exception as exc:
            if type(exc).__name__ == "Unsupported" or is_capacity_error(str(exc)):
                raise Unsupported(str(exc)) from exc
            raise
        return {"answers": engine_answers(request["questions"], response)}
