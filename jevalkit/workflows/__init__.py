"""End-to-end workflows: a model answers typed questions, code makes the decisions, scores compare actions.

A workflow exposes name, title, policies, default_reference, references, dataset, revision, cases() and
execute(case_id, session) -> {"decisions": {policy: {"actions", "primary_action", "outputs"}}, "details"}.
An action is {"name": str, "arguments": dict}. references[labelset][case_id][policy] holds the reference
decision, or None when the reference has no decision for that case and policy.
"""

import importlib
from pathlib import Path

from ..io import dumps, object_hash

WORKFLOWS = {
    "invoice_processing": "jevalkit.workflows.invoice_processing:Workflow",
    "delivery_support": "jevalkit.workflows.delivery_support:Workflow",
}


def load(name, **options):
    if name not in WORKFLOWS:
        raise ValueError("Unknown workflow: " + name)
    module, cls = WORKFLOWS[name].split(":")
    return getattr(importlib.import_module(module), cls)(**options)


def source_revision(folder):
    """Fingerprint of a workflow's code, so resumed and compared runs used the same decisions."""
    folder = Path(folder)
    return object_hash({str(p.relative_to(folder)): p.read_text(encoding="utf-8")
                        for p in sorted(folder.rglob("*.py")) if "__pycache__" not in p.parts})


def action(name, arguments=None):
    return {"name": name, "arguments": arguments or {}}


def action_key(value):
    return dumps({"name": value["name"], "arguments": value.get("arguments") or {}})
