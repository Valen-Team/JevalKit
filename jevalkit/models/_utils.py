"""Shared model resolution, rubric rendering and author-module loading."""

import copy
import importlib
import importlib.util
import sys
from pathlib import Path
from ..io import dumps, object_hash
from ..paths import data_root
from ..schema import native_answer


def resolve_model(model, revision="main", root=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    destination = data_root(root) / "models" / model.replace("/", "--") / revision
    return Path(snapshot_download(model, revision=revision, local_dir=destination))


def render_criteria(questions):
    """Keep candidate identities/order while making description values textual."""
    out = copy.deepcopy(questions)
    for q in out.values():
        if not isinstance(q["instructions"], str):
            q["instructions"] = dumps(q["instructions"])
        if q["type"] == "choice":
            q["criteria"] = {
                k: k if v is None else v if isinstance(v, str) else dumps(v) for k, v in q["criteria"].items()
            }
        elif q["type"] == "score":
            q["criteria"] = [v if isinstance(v, str) else dumps(v) for v in q["criteria"]]
        elif q["type"] == "noul":
            if not q.get("criteria"):
                q.pop("criteria", None)
            else:
                q["criteria"] = {
                    k: k.capitalize() if v is None else v if isinstance(v, str) else dumps(v)
                    for k, v in q["criteria"].items()
                }
    return out


def render_questions(questions):
    """Runtimes with an instruction-only Noul head receive its full rubric there."""
    out = render_criteria(questions)
    for q in out.values():
        if q["type"] == "noul" and "criteria" in q:
            criteria = q.pop("criteria")
            if criteria:
                q["instructions"] += (
                    "\nTrue criterion: "
                    + str(criteria["true"])
                    + "\nFalse criterion: "
                    + str(criteria["false"])
                )
    return out


def load_module(path):
    path = Path(path).resolve()
    name = "_jeval_native_" + object_hash(str(path))[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def import_author_module(folder, name):
    """Import a module the checkpoint ships, refusing a same-named module from anywhere else."""
    folder = Path(folder).resolve()
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    if folder not in origin.parents:
        raise ImportError(f"{name} was imported from {origin}, not from the checkpoint at {folder}")
    return module


def engine_answers(questions, response):
    """Engines that report Noul as P(true) and other types as candidate probabilities."""
    answers = {}
    for qid, question in questions.items():
        answer = response["answers"][qid]
        if question["type"] == "noul":
            probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
        else:
            probabilities = answer["probabilities"]
        answers[qid] = native_answer(question, probabilities)
    return answers
