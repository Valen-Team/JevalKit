"""Own the lifetime and provenance of explicitly configured local model servers."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from .io import atomic_json
from .models.provenance import inference_artifacts
from .security import is_credential_key


def validate_services(services):
    if not isinstance(services, list):
        raise ValueError("services must be a list")
    for service in services:
        command = service.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
            raise ValueError("Each service command must be a nonempty argv list")
        url = urlsplit(service.get("ready_url", ""))
        if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.username or url.password or url.query:
            raise ValueError("Service readiness must use an explicit local HTTP URL without credentials")
        if service.get("timeout", 900) <= 0:
            raise ValueError("Service startup timeout must be positive")
        env = service.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(v, str) for v in env.values()):
            raise ValueError("Service environment must map names to strings")
        if any(is_credential_key(key) for key in env):
            raise ValueError("Service credentials must be inherited from the environment, never saved in matrix JSON")


def ready(url):
    # A localhost server must not travel through an inherited outbound proxy.
    try:
        with build_opener(ProxyHandler({})).open(url, timeout=2) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, ConnectionError):
        return False


def stop_owned(process):
    # The leader can exit while its model workers remain in this owned group.
    # macOS may report EPERM for an exited/zombie-only group during the
    # existence probe, so treat it like a group that is already gone rather
    # than failing cleanup in a finally block.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while True:
        process.poll()  # Reap the leader before checking its remaining workers.
        try:
            os.killpg(process.pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            return
        time.sleep(.1)


@contextmanager
def managed_services(services, folder):
    validate_services(services)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    processes, streams, evidence = [], [], []
    try:
        for index, service in enumerate(services):
            if ready(service["ready_url"]):
                raise ValueError("Service readiness URL already belongs to a running server")
            stream = (folder / f"service-{index}.log").open("w")
            streams.append(stream)
            process = subprocess.Popen(service["command"], cwd=service.get("cwd"),
                env={**os.environ, **service.get("env", {})}, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            processes.append(process)
            deadline = time.monotonic() + service.get("timeout", 900)
            while not ready(service["ready_url"]):
                if process.poll() is not None:
                    raise RuntimeError(f"Model service {index} exited with code {process.returncode}; inspect {stream.name}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Model service {index} startup timed out; inspect {stream.name}")
                time.sleep(.5)
            artifacts = {}
            for source in service.get("provenance_paths", []):
                path = Path(source).resolve(strict=True)
                if path.is_dir():
                    artifacts[str(path)] = inference_artifacts(path)
                else:
                    from .io import sha256
                    artifacts[str(path)] = sha256(path)
            evidence.append({"command": service["command"], "cwd": service.get("cwd"),
                             "environment_options": service.get("env", {}), "artifacts_sha256": artifacts})
        metadata = {"schema_version": 1, "origin": "managed-local-model-services", "services": evidence}
        try:
            import torch
            metadata["hardware"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        except ImportError:
            metadata["hardware"] = "unknown"
        manifest = folder / "provenance.json"
        atomic_json(manifest, metadata)
        yield manifest
    finally:
        for process in reversed(processes):
            stop_owned(process)
        for stream in streams:
            stream.close()
