"""Run a workflow case by case with a resumable journal, then score decisions against references."""

import fcntl
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..io import atomic_json, dumps, jsonl, load_json, object_hash, sha256
from ..pricing import cost_usd, rates_for
from ..schema import Unsupported
from ..security import is_credential_key
from . import action_key, load
from .session import Session


def run_workflow(workflow_name, model_name, output, *, root=None, model_options=None, limit=None, resume=False,
                 workers=1, dataset_revision=None, model_instance=None):
    from ..config import build_model

    options = model_options or {}
    if any(is_credential_key(key) for key in options):
        raise ValueError("Pass a token_env variable name, never a credential value")
    if workers < 1 or limit is not None and limit <= 0:
        raise ValueError("workers and limit must be positive")
    workflow = load(workflow_name, revision=dataset_revision, root=root)
    cases = workflow.cases()[:limit] if limit else workflow.cases()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another evaluator owns this output directory") from None
        signature = {
            "workflow": workflow.name, "workflow_revision": workflow.revision, "dataset": workflow.dataset,
            "model": model_name, "model_options": options, "selected_cases_sha256": object_hash(cases),
            "limit": limit, "jevalkit": __version__, "python": platform.python_version(),
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Output contains an existing run; use --resume or a new directory")
            if load_json(manifest_path)["signature"] != signature:
                raise ValueError("Resume data, workflow code, model or settings changed")
        model = model_instance or build_model(model_name, root=root, **options)
        runtime = model.metadata()
        if manifest_path.exists() and load_json(manifest_path)["runtime"] != runtime:
            model.close()
            raise ValueError("Resume model weights or runtime changed")
        rates = rates_for(runtime)
        if not manifest_path.exists():
            atomic_json(manifest_path, {"signature": signature, "runtime": runtime, "pricing": rates,
                                        "created_utc": datetime.now(timezone.utc).isoformat()})
        journal = output / "cases.jsonl"
        kept = [row for row in jsonl(journal) if row["status"] != "error"] if journal.exists() else []
        if journal.exists():
            # Resume reruns failed cases instead of keeping them as disagreement.
            journal.write_text("".join(dumps(row) + "\n" for row in kept), encoding="utf-8")
        done = {row["case_id"] for row in kept}
        pending = [case for case in cases if case not in done]
        written = threading.Lock()

        def execute(case_id):
            session, started = Session(model, rates), time.perf_counter()
            row = {"case_id": case_id, "status": "ok"}
            try:
                row.update(workflow.execute(case_id, session))
            except Unsupported as exc:
                row.update(status="unsupported", error=str(exc))
            except Exception as exc:
                row.update(status="error", exception=type(exc).__name__, error=str(exc))
            row.update(wall_time_s=time.perf_counter() - started, **session.totals(), calls=session.calls)
            with written, journal.open("a", encoding="utf-8") as stream:
                stream.write(dumps(row) + "\n")
            return row

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                recent = []
                for count, row in enumerate(pool.map(execute, pending), 1):
                    recent = (recent + [row["status"] == "error"])[-5:]
                    if len(recent) == 5 and all(recent):
                        pool.shutdown(cancel_futures=True)
                        raise RuntimeError("Five consecutive case errors; check the model before resuming")
                    if count % 10 == 0 or count == len(pending):
                        print(dumps({"event": "progress", "workflow": workflow.name, "model": model_name,
                                     "completed": len(done) + count, "planned": len(cases)}), flush=True)
        finally:
            model.close()
        return score_run(output, workflow, root)


def compare(workflow_name, folders, reference=None, root=None):
    """A Markdown table of local runs and any published runs, sorted by exact-action agreement."""
    workflow = load(workflow_name, root=root)
    reference = reference or workflow.default_reference
    rows = []
    for folder in folders:
        report = score_run(folder, workflow, root)
        scores = report["references"][reference]
        runtime = report["runtime"]
        rows.append({"model": report["model"] + (f" ({runtime['model']})" if runtime.get("model") else ""),
                     "source": "local" if report["complete"] else f"local, {report['cases']} cases",
                     "exact_actions": scores["exact_actions"], "primary_action": scores["primary_action"],
                     "mean_cost_usd": report["mean_cost_usd"], "mean_wall_time_s": report["mean_wall_time_s"]})
    if hasattr(workflow, "published") and reference == "consensus":
        rows += [{**run, "source": "published"} for run in workflow.published()]
    rows.sort(key=lambda r: -(r["exact_actions"] or 0))
    fmt = lambda v, f: "–" if v is None else f.format(v)
    lines = [f"{workflow.title}, agreement with the {reference} reference", "",
             "| Model | Source | Exact actions | Primary action | Cost / case (USD) | Time / case (s) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['model']} | {r['source']} | {fmt(r['exact_actions'], '{:.1%}')} | "
              f"{fmt(r['primary_action'], '{:.1%}')} | {fmt(r['mean_cost_usd'], '{:.5f}')} | "
              f"{fmt(r['mean_wall_time_s'], '{:.2f}')} |" for r in rows]
    return "\n".join(lines) + "\n"


def agreement(reference, decision):
    """Exact action-set and primary-action agreement; None where the reference has no answer."""
    if reference is None:
        return None, None
    if decision is None:
        return False, False if reference["primary_action"] is not None else None
    exact = {action_key(a) for a in reference["actions"]} == {action_key(a) for a in decision["actions"]}
    if reference["primary_action"] is None:
        return exact, None
    primary = decision["primary_action"] is not None and \
        action_key(reference["primary_action"]) == action_key(decision["primary_action"])
    return exact, primary


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def score_run(output, workflow=None, root=None):
    """Failed cases stay in every denominator and count as disagreement."""
    output = Path(output)
    manifest = load_json(output / "manifest.json")
    signature = manifest["signature"]
    workflow = workflow or load(signature["workflow"], revision=signature["dataset"]["revision"], root=root)
    if workflow.revision != signature["workflow_revision"] or workflow.dataset != signature["dataset"]:
        raise ValueError("Workflow code or dataset changed since this run")
    rows = list(jsonl(output / "cases.jsonl"))
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case in journal")
    references = {}
    for labelset, labels in workflow.references.items():
        by_policy = {}
        for policy in workflow.policies:
            pairs = [agreement(labels[r["case_id"]][policy], r.get("decisions", {}).get(policy)) for r in rows]
            by_policy[policy] = {
                "exact_actions": mean(p[0] for p in pairs), "primary_action": mean(p[1] for p in pairs),
                "exact_denominator": sum(p[0] is not None for p in pairs),
                "primary_denominator": sum(p[1] is not None for p in pairs),
            }
        references[labelset] = {
            "exact_actions": mean(v["exact_actions"] for v in by_policy.values()),
            "primary_action": mean(v["primary_action"] for v in by_policy.values()),
            "policies": by_policy,
        }
    # Price recorded tokens with the current table, so a rate added after the run still applies.
    rates = rates_for(manifest["runtime"])

    def case_cost(row):
        costs = [cost_usd(rates, call.get("input_tokens"), call.get("output_tokens")) for call in row["calls"]]
        return sum(costs) if None not in costs else None

    costs = [case_cost(r) for r in rows]
    report = {
        "schema_version": 1, "workflow": workflow.name, "model": signature["model"], "runtime": manifest["runtime"],
        "cases": len(rows), "selected_cases": len(workflow.cases()[:signature["limit"]] if signature["limit"]
                                                  else workflow.cases()),
        "complete": signature["limit"] is None and len(rows) == len(workflow.cases()),
        "status": {s: sum(r["status"] == s for r in rows) for s in ("ok", "unsupported", "error")},
        "default_reference": workflow.default_reference, "references": references,
        "pricing": rates,
        "mean_cost_usd": sum(costs) / len(costs) if costs and None not in costs else None,
        "mean_wall_time_s": mean(r["wall_time_s"] for r in rows),
        "mean_calls": mean(r["call_count"] for r in rows),
        "journal_sha256": sha256(output / "cases.jsonl"), "manifest_sha256": sha256(output / "manifest.json"),
    }
    atomic_json(output / "scores.json", report)
    return report
