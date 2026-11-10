"""Adapters for author engines shipped inside checkpoints, exercised with synthetic engines on CPU.

No model weights are loaded and no benchmark scores are produced.
"""

import sys
import textwrap

import pytest

from jevalkit.config import MODELS
from jevalkit.datasets.decision_index_vision import configure_model_options
from jevalkit.models._utils import import_author_module
from jevalkit.schema import Unsupported

AUTHOR_MODULES = ("amos", "amos.serve", "amos.decoder", "d3_engine", "d3_runtime", "inference",
                  "inference.engine", "rbdash_decision")

QUESTIONS = {
    "route": {"type": "choice", "instructions": "Route it", "criteria": {"billing": "Money", "support": None}},
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "severity": {"type": "score", "instructions": "Rate it", "criteria": ["low", "mid", "high"]},
}


@pytest.fixture(autouse=True)
def fresh_author_modules(monkeypatch):
    """Each test gets its own checkpoint folder; never reuse a same-named module from another one."""
    for name in AUTHOR_MODULES:
        sys.modules.pop(name, None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in AUTHOR_MODULES:
        sys.modules.pop(name, None)


def write(folder, files):
    for name, body in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return folder


def request(**extra):
    return {"state": "A customer asks about an invoice", "questions": QUESTIONS, **extra}


D3_ENGINE = """
    class Decision:
        max_length = 32768

    class D3Engine:
        def __init__(self, model, device, **options):
            self.model, self.device, self.options = model, device, options
            self.provenance = {"kind": "fixture", "device": device}
            self.decision = Decision()
            self.calls = []

        def __call__(self, state, questions, images=None, videos=None):
            self.calls.append({"state": state, "questions": questions, "images": images})
            if "too long" in state:
                raise ValueError("question q: 40000 tokens is over the limit")
            return {"answers": {
                "route": {"probabilities": {"billing": 0.25, "support": 0.75}},
                "urgent": {"noul": 0.8},
                "severity": {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}},
            }}, {}

        def close(self):
            self.closed = True
"""


def test_new_models_are_registered_with_their_published_checkpoints():
    expected = {
        "Amos-1-Large": ("Amos", "mukaddamzaid/amos-1-large"),
        "vLLM-SR-d3-turbo": ("vLLM-SR-d3", "vllm-sr/d3-turbo"),
        "WeMM-JEV": ("WeMM-JEV", "Rover00/WeMM-JEV"),
        "Xing-VL2.0-Decision": ("Xing-VL", "RBDash-Team/Xing-VL2.0-Decision"),
        **{f"Valen-Decision-1.5-{s}": ("Valen", f"Valen-Team/Valen-Decision-1.5-{s}")
           for s in ("0.8B", "2B", "4B", "9B", "27B")},
    }
    for name, (family, checkpoint) in expected.items():
        spec = MODELS[name]
        assert (spec.family, spec.model, spec.transport) == (family, checkpoint, "native")
        assert spec.source_url == "https://huggingface.co/" + checkpoint
    families = {s.family for s in MODELS.values() if s.family not in {"baseline", "LLM"}}
    assert len(families) == 24


def test_valen_decision_15_uses_the_board_image_budget():
    assert configure_model_options("Valen-Decision-1.5-27B", {}) == {"max_image_pixels": 1638400}
    with pytest.raises(ValueError, match="1638400"):
        configure_model_options("Valen-Decision-1.5-0.8B", {"max_image_pixels": 1048576})


def test_author_modules_must_come_from_the_checkpoint(tmp_path):
    first = write(tmp_path / "first", {"d3_runtime.py": "SCORE_LEVELS = (2, 10)\n"})
    second = write(tmp_path / "second", {"d3_runtime.py": "SCORE_LEVELS = (2, 5)\n"})
    assert import_author_module(first, "d3_runtime").SCORE_LEVELS == (2, 10)
    with pytest.raises(ImportError, match="not from the checkpoint"):
        import_author_module(second, "d3_runtime")


def test_d3_passes_questions_and_converts_answers(tmp_path):
    from jevalkit.models.d3 import D3Model

    folder = write(tmp_path / "d3", {"d3_engine.py": D3_ENGINE, "d3_runtime.py": "SCORE_LEVELS = (2, 10)\n"})
    model = D3Model(str(folder), device="cpu", root=tmp_path)
    assert model.engine.device == "cpu"
    assert model.info["max_length"] == 32768 and model.info["score_levels"] == [2, 10]
    answers = model.predict(request())["answers"]
    assert answers["urgent"] == {"type": "noul", "noul": 0.8}
    assert answers["route"]["choice"] == "support"
    assert answers["severity"]["score"] == pytest.approx(1.3)
    sent = model.engine.calls[0]
    assert sent["images"] is None and sent["questions"]["route"]["criteria"] == {"billing": "Money", "support": None}
    with pytest.raises(Unsupported, match="over the limit"):
        model.predict({**request(), "state": "too long"})
    eleven = {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(11)]}
    with pytest.raises(Unsupported, match="2 to 10 score levels"):
        model.predict({"state": "x", "questions": {"s": eleven}})


def test_d3_and_wemm_reject_video_without_dropping_it(tmp_path):
    from jevalkit.models.d3 import D3Model

    folder = write(tmp_path / "d3", {"d3_engine.py": D3_ENGINE, "d3_runtime.py": "SCORE_LEVELS = (2, 10)\n"})
    (tmp_path / "clip.mp4").write_bytes(b"video")
    state = {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "clip.mp4"}}]}]}
    model = D3Model(str(folder), device="cpu", root=tmp_path)
    with pytest.raises(Unsupported, match="video"):
        model.predict({"state": state, "questions": QUESTIONS}, media_root=tmp_path)
    assert model.engine.calls == []


def test_wemm_limits_score_levels_and_closes_its_engine(tmp_path):
    from jevalkit.models.wemm_jev import WeMMJevModel

    folder = write(tmp_path / "wemm", {
        "inference/__init__.py": "from .engine import D3Engine\n",
        "inference/engine.py": D3_ENGINE.replace("return {\"answers\": {", "return {\"answers\": {\"echo\": 1, "),
    })
    model = WeMMJevModel(str(folder), device="cpu", root=tmp_path)
    assert model.info["provenance"] == {"kind": "fixture", "device": "cpu"}
    assert model.predict(request())["answers"]["echo"] == 1
    eleven = {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(11)]}
    with pytest.raises(Unsupported, match="at most 10 score levels"):
        model.predict({"state": "x", "questions": {"s": eleven}})
    engine = model.engine
    model.close()
    assert engine.closed and not hasattr(model, "engine")


def test_xing_scores_each_question_in_candidate_order(tmp_path):
    from jevalkit.models.xing_vl import XingVLModel

    folder = write(tmp_path / "xing", {"rbdash_decision.py": """
        MAX_OPTIONS = 3
        MAX_IMAGE_PIXELS = 1638400
        MAX_IMAGES = 1
        CALLS = []

        def normalise_options(options, kind):
            if kind == "noul":
                return ["false", "true"], ["False", "True"]
            if isinstance(options, dict):
                return list(options), [str(v if v is not None else k) for k, v in options.items()]
            return [str(i) for i in range(len(options))], list(options)

        class RBDashDecision:
            temperature = 0.97

            @classmethod
            def from_pretrained(cls, path, device=None, **kwargs):
                engine = cls()
                engine.device = device
                return engine

            def probabilities(self, state, question, descriptions, images=()):
                CALLS.append((question, list(descriptions), list(images)))
                if question == "huge":
                    raise ValueError("prompt has 99999 tokens; nothing is truncated")
                n = len(descriptions)
                return [(i + 1) / (n * (n + 1) / 2) for i in range(n)]
    """})
    model = XingVLModel(str(folder), device="cpu", root=tmp_path)
    assert model.engine.device == "cpu" and model.info["temperature"] == 0.97
    answers = model.predict(request())["answers"]
    assert answers["route"]["probabilities"] == pytest.approx({"billing": 1 / 3, "support": 2 / 3})
    assert answers["urgent"]["noul"] == pytest.approx(2 / 3)
    assert model.rd.CALLS[0] == ("Route it", ["Money", "support"], [])
    four = {"type": "choice", "instructions": "Pick", "criteria": {k: None for k in "abcd"}}
    with pytest.raises(Unsupported, match="at most 3 options"):
        model.predict({"state": "x", "questions": {"q": four}})
    with pytest.raises(Unsupported, match="nothing is truncated"):
        model.predict({"state": "x", "questions": {"q": {**QUESTIONS["urgent"], "instructions": "huge"}}})
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"png")
    two = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": name}} for name in ("a.png", "b.png")]}]}
    with pytest.raises(Unsupported, match="at most 1 images"):
        model.predict({"state": two, "questions": QUESTIONS}, media_root=tmp_path)


def test_amos_sends_images_inline_and_maps_engine_rejections(tmp_path):
    from jevalkit.models.amos import AmosModel

    folder = write(tmp_path / "amos", {
        "serve/amos/__init__.py": "",
        "serve/amos/decoder.py": "IMAGE_PIXELS = 1048576\n",
        "serve/amos/serve.py": """
            class BadRequest(ValueError):
                pass

            class Engine:
                card = {"base": "fixture", "max_len": None}

                def __init__(self, run, device, name, max_len, rotations):
                    self.args = (run, device, name, max_len, rotations)
                    self.payloads = []

                def __call__(self, payload):
                    self.payloads.append(payload)
                    if payload["state"] == "long":
                        raise BadRequest("maximum context length is 131072 tokens")
                    if payload["state"] == "broken":
                        raise BadRequest("malformed request")
                    return {"answers": {
                        "route": {"probabilities": {"billing": 0.9, "support": 0.1}},
                        "urgent": {"noul": 0.1},
                        "severity": {"probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}},
                    }}

            def load_engine(run, device, name=None, max_len=None, rotations=1):
                return Engine(run, device, name, max_len, rotations)
        """,
    })
    model = AmosModel(str(folder), device="cpu", root=tmp_path)
    assert model.engine.args[1:] == ("cpu", None, 131072, 1)
    assert model.info["image_pixels"] == 1048576 and model.info["rotations"] == 1
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    state = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "See photo"}, {"type": "image_url", "image_url": {"url": "photo.png"}}]}]}
    answers = model.predict({"state": state, "questions": QUESTIONS}, media_root=tmp_path)["answers"]
    assert answers["route"]["choice"] == "billing" and answers["urgent"]["noul"] == pytest.approx(0.1)
    assert answers["severity"]["score"] == pytest.approx(2)
    assert model.engine.payloads[0]["images"] == ["iVBORw=="]
    with pytest.raises(Unsupported, match="maximum context length"):
        model.predict({**request(), "state": "long"})
    with pytest.raises(ValueError, match="malformed") as raised:
        model.predict({**request(), "state": "broken"})
    assert not isinstance(raised.value, Unsupported)
