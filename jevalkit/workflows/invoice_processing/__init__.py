"""TypeSafe's invoice-processing workflow from WorkflowEvals, answered by any JevalKit model.

upstream/ is evals/invoice_processing/structured/ from typesafe-ai/WorkflowEvals at
0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb (Apache-2.0, see upstream/LICENSE), with only its
typesafe_sdk import pointed at jevalkit.workflows.sdk_types. Cases and the OpenAI, Anthropic and
consensus reference decisions come from the Hugging Face dataset below.
"""

import json
from pathlib import Path

from .. import action, source_revision
from ...io import load_json, sha256
from ...paths import data_root
from ..sdk_types import from_native

REPO = "typesafeAI/evalsafe-invoice-processing"


class Client:
    """The SystemOneLike client upstream calls; the model argument is fixed by the session."""

    def __init__(self, session):
        self.session = session

    def system_one(self, model, state, questions):
        response = self.session.ask(state, {qid: q.wire() for qid, q in questions.items()})
        return from_native(response["answers"], response.get("usage") or {}, str(response.get("model") or ""))


def reference(label, policies):
    decisions = {d["policy_id"]: d for d in label["decisions"]}
    out = {}
    for policy in policies:
        d = decisions.get(policy)
        out[policy] = None if d is None or d["status"] != "ok" else {
            "actions": [action(a["name"], json.loads(a["arguments_json"])) for a in d["actions"]],
            "primary_action": action(d["primary_action"]["name"], json.loads(d["primary_action"]["arguments_json"]))
            if d["primary_action"] else None,
        }
    return out


class Workflow:
    name, title, default_reference = "invoice_processing", "Invoice processing", "consensus"

    def __init__(self, revision=None, root=None):
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as parquet

        cache = data_root(root) / "cache" / "hf" / "hub"
        manifest_path = Path(hf_hub_download(REPO, "dataset.json", repo_type="dataset", revision=revision or "main",
                                             cache_dir=cache))
        commit = manifest_path.parent.name
        cases_path = hf_hub_download(REPO, "data/cases.parquet", repo_type="dataset", revision=commit, cache_dir=cache)
        self.manifest = load_json(manifest_path)
        rows = parquet.read_table(cases_path).to_pylist()
        self.inputs = {row["case_id"]: json.loads(row["input_json"]) for row in rows}
        self.policies = [p["policy_id"] for p in self.manifest["policies"]]
        self.references = {role: {row["case_id"]: reference(row[role], self.policies) for row in rows}
                           for role in self.manifest["labelsets"]}
        self.dataset = {"source": "https://huggingface.co/datasets/" + REPO, "revision": commit,
                        "cases_sha256": sha256(cases_path), "license": self.manifest["license"],
                        "labels": "model-generated references"}
        self.revision = source_revision(Path(__file__).parent)

    def cases(self):
        return list(self.inputs)

    def published(self):
        """TypeSafe's published runs, scored on all cases against the consensus reference."""
        out = []
        for run in self.manifest["runs"].values():
            scores = {s["metric_id"]: s["value"] for s in run["scores"]}
            model = run["model"]
            out.append({"model": f"{model['provider']}:{model['name']}", "effort": model["reasoning_effort"],
                        "reference": "consensus", "exact_actions": scores["accuracy"],
                        "primary_action": scores["primary_match"], "mean_cost_usd": run["mean_cost_usd"],
                        "mean_wall_time_s": run["mean_wall_time_s"], "cases": run["cases"]})
        return out

    def execute(self, case_id, session):
        from .upstream.decompose import run_case
        from .upstream.policy import PROFILES
        from .upstream.state import Packet

        result = run_case(Packet.model_validate(self.inputs[case_id]), Client(session), "",
                          profiles=[PROFILES[p] for p in self.policies])
        decisions = {
            name: {
                "actions": [action(a.value) for a in d.actions],
                "primary_action": action(d.primary_action.value),
                "outputs": d.model_dump(mode="json", exclude={"policy", "actions", "primary_action", "trace"}),
            }
            for name, d in result.decisions.items()
        }
        return {"decisions": decisions,
                "details": {"skipped_questions": result.question_plan.skipped,
                            "findings": result.findings.model_dump(mode="json") if result.findings else None}}
