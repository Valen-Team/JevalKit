import json
import signal
import socket
import sys
from types import SimpleNamespace

import pytest

from jevalkit.services import managed_services, ready, stop_owned, validate_services


def unused_local_url():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{listener.getsockname()[1]}/health"


def test_managed_server_becomes_ready_records_weights_and_stops(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    url = unused_local_url()
    port = url.split(":")[-1].split("/")[0]
    code = """from http.server import HTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200)
  self.end_headers()
 def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(__import__('sys').argv[1])), H).serve_forever()
"""
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "head.pt").write_text("fixture")
    service = {"command": [sys.executable, "-c", code, port], "ready_url": url,
               "timeout": 10, "provenance_paths": [str(weights)]}
    with managed_services([service], tmp_path / "server") as manifest:
        assert ready(url)
        metadata = json.loads(manifest.read_text())
        assert metadata["hardware"] == "cpu"
        assert set(metadata["services"][0]["artifacts_sha256"][str(weights)]) == {"head.pt"}
    assert not ready(url)


def test_stop_owned_tolerates_macos_permission_error_for_exiting_group(monkeypatch):
    calls = []

    class ExitedProcess:
        pid = 12345

        def poll(self):
            return 0

    def killpg(pid, signum):
        calls.append((pid, signum))
        if signum == 0:
            raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("jevalkit.services.os.killpg", killpg)
    stop_owned(ExitedProcess())

    assert calls == [(12345, signal.SIGTERM), (12345, 0)]


def test_failed_model_startup_is_not_mistaken_for_ready(tmp_path):
    service = {"command": [sys.executable, "-c", "raise SystemExit(7)"],
               "ready_url": unused_local_url(), "timeout": 10}
    with pytest.raises(RuntimeError, match="exited with code 7"):
        with managed_services([service], tmp_path):
            pytest.fail("Failed startup must never run inference")


@pytest.mark.parametrize("change", [{"command": "shell command"},
                                   {"ready_url": "https://remote.example/health"},
                                   {"env": {"API_KEY": "literal-secret"}}])
def test_service_contract_rejects_ambiguous_commands_or_saved_credentials(change):
    spec = {"command": ["python", "server.py"], "ready_url": "http://127.0.0.1:8000/health", **change}
    with pytest.raises(ValueError):
        validate_services([spec])


def test_service_tokenizer_and_token_limits_are_not_credentials():
    validate_services([{"command": ["python", "server.py"], "ready_url": "http://127.0.0.1:8000/health",
                        "env": {"TOKENIZER": "/weights", "MAX_STATE_TOKENS": "32000"}}])
