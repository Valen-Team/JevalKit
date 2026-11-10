"""Native WeMM-JEV adapter: the author's ``inference.D3Engine`` shipped with the checkpoint."""

import json

from .base import NativeRuntime
from ._utils import import_author_module
from .media import state_parts
from ..schema import Unsupported, is_capacity_error

MAX_SCORE_LEVELS = 10


class WeMMJevModel(NativeRuntime):
    """WeMM-JEV for Choice, Noul and Score (2 to 10 levels) over text and images."""

    modalities = frozenset({"text", "image"})

    def __init__(self, model, **options):
        super().__init__(model, **options)
        inference = import_author_module(self.folder, "inference")
        self.engine = inference.D3Engine(model=str(self.folder), device=self.device, **self.options)
        self.info.update(
            provenance=json.loads(json.dumps(self.engine.provenance, default=str)),
            max_score_levels=MAX_SCORE_LEVELS,
        )

    def predict(self, request, *, media_root=None, assets=None):
        state, images, videos = state_parts(request["state"], media_root)
        if videos:
            raise Unsupported("WeMM-JEV video input is not enabled in this adapter; all evidence was retained")
        questions = {}
        for qid, q in request["questions"].items():
            if q["type"] == "score" and len(q["criteria"]) > MAX_SCORE_LEVELS:
                raise Unsupported(
                    f"WeMM-JEV supports at most {MAX_SCORE_LEVELS} score levels; all levels were retained"
                )
            questions[qid] = {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
        try:
            response, _ = self.engine(state=state, questions=questions, images=images or None)
        except Exception as exc:
            if type(exc).__name__ == "Unsupported" or is_capacity_error(str(exc)):
                raise Unsupported(str(exc)) from exc
            raise
        return {"answers": response["answers"]}

    def close(self):
        if hasattr(self, "engine"):
            self.engine.close()
        super().close()
