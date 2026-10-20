"""Evaluate a model/benchmark matrix sequentially with one isolated process per model."""
import subprocess
import sys
from pathlib import Path

from .config import MODELS
from .datasets import BENCHMARKS
from .io import atomic_json, dumps, load_json, object_hash
from .security import is_credential_key


def matrix_commands(config, root=None):
    if config.get("schema_version") != 1 or not isinstance(config.get("runs"), list) or not config["runs"]:
        raise ValueError("Matrix requires schema_version=1 and a nonempty runs list")
    commands = []
    for index, run in enumerate(config["runs"]):
        if run.get("model") not in MODELS:
            raise ValueError(f"Matrix run {index}: unknown model")
        if "services" in run:
            from .services import validate_services
            validate_services(run["services"])
        mode = run.get("mode", "run")
        if mode not in {"run", "check"}:
            raise ValueError("Matrix mode must be run or check")
        data = run.get("data", config.get("data"))
        if mode == "run" and (not isinstance(data, list) or not data or set(data) - set(BENCHMARKS)):
            raise ValueError(f"Matrix run {index}: select registered benchmarks")
        output = run.get("output", config.get("output"))
        if not output:
            raise ValueError(f"Matrix run {index}: output directory is required")
        python = run.get("python", sys.executable)
        if not isinstance(python, str) or not python:
            raise ValueError("Matrix python must name an interpreter")
        command = [python, "-m", "jevalkit", mode, "--model", run["model"]]
        if mode == "run":
            command.extend(["--data", *data, "--output", str(output)])
            rank, world_size = run.get("rank", 0), run.get("world_size", 1)
            if (type(rank) is not int or type(world_size) is not int
                    or world_size < 1 or not 0 <= rank < world_size):
                raise ValueError("Matrix shards require integer 0 <= rank < world_size")
            if "rank" in run or "world_size" in run:
                command.extend(["--rank", str(rank), "--world-size", str(world_size)])
        else:
            if any(k in run for k in ("limit", "warmup", "resume", "rank", "world_size")):
                raise ValueError("Check mode does not accept benchmark sampling/resume settings")
            command.extend(["--output", str(Path(output) / run["model"] / "conformance.json")])
        if root:
            command.extend(["--root", str(root)])
        for key, value in run.get("options", {}).items():
            if is_credential_key(key):
                raise ValueError("Matrix credentials must be supplied through token_env, never literal values")
            command.extend(["--option", key + "=" + dumps(value)])
        if run.get("limit") is not None:
            if not isinstance(run["limit"], int) or run["limit"] <= 0:
                raise ValueError("Matrix limits must be positive integers")
            command.extend(["--limit", str(run["limit"])])
        if run.get("resume", False):
            command.append("--resume")
        if "warmup" in run:
            command.extend(["--warmup", str(run["warmup"])])
        commands.append(command)
    return commands


def run_matrix(path, root=None, status=None):
    config = load_json(path)
    commands = matrix_commands(config, root)
    status = Path(status or Path(path).with_suffix(".status.json"))
    report = {"schema_version": 1, "config_sha256": object_hash(config), "runs": [],
              "policy": "Sequential isolated subprocesses; failures are retained and independent models continue"}
    for index, command in enumerate(commands):
        row = {"index": index, "model": config["runs"][index]["model"], "command": command, "state": "running"}
        report["runs"].append(row)
        atomic_json(status, report)
        print(dumps({"event": "matrix_run_started", "index": index, "model": row["model"]}), flush=True)
        try:
            if config["runs"][index].get("services"):
                from .services import managed_services
                with managed_services(config["runs"][index]["services"], status.parent / f"services-{index}") as manifest:
                    row["command"] = [*command, "--option", "server_manifest=" + dumps(str(manifest.resolve()))]
                    atomic_json(status, report)
                    result = subprocess.run(row["command"], check=False)
            else:
                result = subprocess.run(command, check=False)
            row.update(state="succeeded" if result.returncode == 0 else "failed", exit_code=result.returncode)
        except Exception as exc:
            row.update(state="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        atomic_json(status, report)
        print(dumps({"event": "matrix_run_finished", "index": index, "model": row["model"],
                     "exit_code": row["exit_code"]}), flush=True)
    report["complete"] = all(r["state"] == "succeeded" for r in report["runs"])
    atomic_json(status, report)
    return report
