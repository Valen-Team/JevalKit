import pytest

from jevalkit.cli import options
from jevalkit.config import build_model
from jevalkit.matrix import matrix_commands
from jevalkit.runner import run
from jevalkit.security import is_credential_key
from jevalkit.services import validate_services


@pytest.mark.parametrize(
    "key",
    ["token", "API_KEY", "password", "secret", "access_token", "hf_token", "service_api_key"],
)
def test_credential_keys_include_namespaced_variants(key):
    assert is_credential_key(key)


@pytest.mark.parametrize("key", ["token_env", "tokenizer", "max_state_tokens", "secretary"])
def test_noncredential_token_names_remain_available(key):
    assert not is_credential_key(key)


def test_literal_namespaced_credentials_are_rejected_at_every_configuration_boundary(tmp_path):
    with pytest.raises(ValueError, match="token_env"):
        options(["access_token=literal-secret"])
    with pytest.raises(ValueError, match="token_env"):
        build_model("Uniform", hf_token="literal-secret")
    with pytest.raises(ValueError, match="token_env"):
        run(
            "Uniform",
            "missing-benchmark",
            tmp_path / "unused-output",
            model_options={"access_token": "literal-secret"},
        )
    assert not (tmp_path / "unused-output").exists()
    with pytest.raises(ValueError, match="token_env"):
        matrix_commands(
            {
                "schema_version": 1,
                "data": ["JevBench"],
                "output": "results",
                "runs": [{"model": "Uniform", "options": {"service_api_key": "literal-secret"}}],
            }
        )
    with pytest.raises(ValueError, match="inherited"):
        validate_services(
            [
                {
                    "command": ["python", "server.py"],
                    "ready_url": "http://127.0.0.1:8000/health",
                    "env": {"hf_token": "literal-secret"},
                }
            ]
        )
