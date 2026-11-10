"""Native Xing-VL adapter: the author's ``rbdash_decision`` readout shipped with the checkpoint."""

from .base import NativeRuntime
from ._utils import import_author_module
from .media import state_parts
from ..schema import Unsupported, candidates, is_capacity_error, native_answer


class XingVLModel(NativeRuntime):
    """Xing-VL2.0-Decision: one forward pass per question over the author's 255-way readout."""

    modalities = frozenset({"text", "image"})

    def __init__(self, model, **options):
        super().__init__(model, **options)
        self.rd = import_author_module(self.folder, "rbdash_decision")
        self.engine = self.rd.RBDashDecision.from_pretrained(str(self.folder), device=self.device, **self.options)
        self.info.update(
            temperature=self.engine.temperature,
            max_pixels=self.rd.MAX_IMAGE_PIXELS,
            max_images=self.rd.MAX_IMAGES,
            max_options=self.rd.MAX_OPTIONS,
        )

    def predict(self, request, *, media_root=None, assets=None):
        state, images, videos = state_parts(request["state"], media_root)
        if videos:
            raise Unsupported("Xing-VL2.0-Decision has no video input; all evidence was retained")
        if len(images) > self.rd.MAX_IMAGES:
            raise Unsupported(f"Xing-VL2.0-Decision accepts at most {self.rd.MAX_IMAGES} images per request")
        answers = {}
        for qid, q in request["questions"].items():
            keys = list(candidates(q))
            if len(keys) > self.rd.MAX_OPTIONS:
                raise Unsupported(
                    f"Xing-VL2.0-Decision supports at most {self.rd.MAX_OPTIONS} options; none were removed"
                )
            _, descriptions = self.rd.normalise_options(q.get("criteria"), q["type"])
            try:
                probabilities = self.engine.probabilities(state, q["instructions"], descriptions, images)
            except ValueError as exc:
                if "nothing is truncated" in str(exc) or is_capacity_error(str(exc)):
                    raise Unsupported(str(exc)) from exc
                raise
            answers[qid] = native_answer(q, dict(zip(keys, probabilities, strict=True)))
        return {"answers": answers}
