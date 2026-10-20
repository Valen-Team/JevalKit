"""Shared checks that keep literal credentials out of persisted configuration."""


_CREDENTIAL_NAMES = ("token", "api_key", "password", "secret")
_NAMESPACED_CREDENTIAL_SUFFIXES = tuple("_" + name for name in _CREDENTIAL_NAMES)


def is_credential_key(key):
    """Recognize direct and namespaced credential keys without matching tokenizers."""
    normalized = key.lower()
    return normalized in _CREDENTIAL_NAMES or normalized.endswith(_NAMESPACED_CREDENTIAL_SUFFIXES)
