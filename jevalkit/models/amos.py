"""Native Amos 1 adapter: the checkpoint's own serving engine, called in-process."""

import base64
import json
from pathlib import Path

from .base import NativeRuntime
from ._utils import engine_answers, import_author_module
from .media import state_parts
from ..schema import Unsupported


class AmosModel(NativeRuntime):
    """Amos 1 through ``amos.serve.load_engine`` from the checkpoint's ``serve/`` folder, without HTTP.

    Defaults follow the author's serving command (``--max-len 131072 --rotations 1 --strict``) and the
    engine's own image budget. Out-of-memory on one GPU is recorded as unsupported with its own message.
    """

    modalities = frozenset({"text", "image"})

    def __init__(self, model, max_len=131072, rotations=1, **options):
        super().__init__(model, **options)
        serve = self.folder / "serve"
        decoder = import_author_module(serve, "amos.decoder")
        server = import_author_module(serve, "amos.serve")
        self.bad_request = server.BadRequest
        self.engine = server.load_engine(str(self.folder), self.device, None, int(max_len), int(rotations))
        self.info.update(
            max_len=int(max_len),
            rotations=int(rotations),
            image_pixels=decoder.IMAGE_PIXELS,
            card=json.loads(json.dumps(self.engine.card, default=str)),
        )

    def predict(self, request, *, media_root=None, assets=None):
        import torch

        state, images, videos = state_parts(request["state"], media_root)
        if videos:
            raise Unsupported("Amos 1 has no video input; all evidence was retained")
        questions = {
            qid: {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
            for qid, q in request["questions"].items()
        }
        payload = {
            "state": state,
            "questions": questions,
            "images": [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images],
        }
        try:
            response = self.engine(payload)
        except self.bad_request as exc:
            message = str(exc)
            if message.startswith("maximum context length") or "criteria must" in message:
                raise Unsupported(message) from exc
            raise
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise Unsupported(
                "out of GPU memory on one device (input retained, not truncated): " + str(exc).splitlines()[0]
            ) from exc
        return {"answers": engine_answers(request["questions"], response)}
