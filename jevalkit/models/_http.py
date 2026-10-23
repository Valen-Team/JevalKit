"""Jev-compatible HTTP and the native OpenAI Decisions API (not Responses)."""

import base64
import copy
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import DecisionModel
from ..io import dumps
from ..schema import Unsupported, is_message_state


def media_data_url(url, root):
    if url.startswith(("https://", "http://", "data:")):
        return url
    path = Path(root or ".") / url
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def api_state(state, root, openai=False):
    if isinstance(state, str):
        return state
    if not is_message_state(state):
        return dumps(state)
    messages = []
    for message in copy.deepcopy(state["messages"]):
        if message["role"] != "user":
            raise Unsupported("The decision input supports user messages only")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "video_url":
                raise Unsupported("This API does not support native video inputs")
            if part["type"] == "text":
                parts.append({"type": "input_text", "text": part["text"]} if openai else part)
            elif part["type"] == "image_url":
                url = media_data_url(part["image_url"]["url"], root)
                parts.append(
                    {"type": "input_image", "image_url": url}
                    if openai
                    else {"type": "image_url", "image_url": {"url": url}}
                )
            else:
                raise Unsupported("Unknown media type: " + part["type"])
        messages.append({"role": "user", "content": parts, **({"type": "message"} if openai else {})})
    return messages if openai else {"messages": messages}


class HttpDecisionModel(DecisionModel):
    modalities = frozenset({"text", "image"})
    retryable_statuses = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        endpoint,
        model,
        token_env="JEVAL_API_KEY",
        timeout=120,
        retries=3,
        media_format="messages",
        server_manifest=None,
        **options,
    ):
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("An explicit HTTP(S) endpoint is required")
        self.endpoint, self.model, self.token_env = endpoint, model, token_env
        self.timeout, self.retries = float(timeout), int(retries)
        if media_format not in {"messages", "text", "top_level", "state_image"}:
            raise ValueError("Unknown decision API media format")
        self.media_format = media_format
        self.server_manifest = server_manifest

    def unsupported_http_reason(self, error):
        """Return a safe capacity reason for a known server contract, if any."""
        return None

    def request_headers(self):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.token_env)
        if token:
            headers["Authorization"] = "Bearer " + token
        return headers

    def post(self, payload):
        headers = self.request_headers()
        for attempt in range(self.retries + 1):
            try:
                req = Request(self.endpoint, data=dumps(payload).encode(), headers=headers, method="POST")
                with urlopen(req, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                reason = self.unsupported_http_reason(exc)
                if reason is not None:
                    raise Unsupported(reason) from None
                # Never persist response text that may echo authorization or input data.
                if exc.code not in self.retryable_statuses or attempt == self.retries:
                    raise RuntimeError(f"Decision API HTTP {exc.code}") from None
                retry_after = exc.headers.get("Retry-After", "")
                delay = min(float(retry_after), 30) if retry_after.isdigit() else min(2**attempt, 30)
            except (URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("Decision API connection failed after retries") from None
                delay = min(2**attempt, 30)
            time.sleep(delay)

    def predict(self, request, *, media_root=None, assets=None):
        payload = {"model": self.model, "questions": copy.deepcopy(request["questions"])}
        if self.media_format in {"top_level", "state_image"}:
            from .media import state_parts

            text, images, videos = state_parts(request["state"], media_root)
            if videos:
                raise Unsupported("No verified video transport for this HTTP adapter")
            payload["state"] = text
            if images:
                encoded = [media_data_url(p, media_root) for p in images]
                if self.media_format == "state_image":
                    if len(encoded) != 1:
                        raise Unsupported("OpenJev's published shim supports one screenshot per request")
                    payload["state"] = {"text": text, "image": encoded[0]}
                else:
                    payload["images"] = encoded
        elif self.media_format == "text":
            from .media import state_parts

            text, images, videos = state_parts(request["state"], media_root)
            if images or videos:
                raise Unsupported("This HTTP release supports text only")
            payload["state"] = text
        else:
            payload["state"] = api_state(request["state"], media_root)
        return self.post(payload)

    def metadata(self):
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(self.endpoint)
        # Credentials and query parameters must not become run provenance.
        metadata = {
            "adapter": type(self).__name__,
            "model": self.model,
            "endpoint": urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, "", "")),
            "token_env": self.token_env,
            "timeout": self.timeout,
            "retries": self.retries,
            "media_format": self.media_format,
        }
        if self.server_manifest:
            from ..io import load_json, sha256

            server = load_json(self.server_manifest)
            if server.get("origin") != "managed-local-model-services" or server.get("schema_version") != 1:
                raise ValueError("Invalid managed service provenance")
            metadata.update(
                server=server,
                server_manifest_sha256=sha256(self.server_manifest),
                hardware=server["hardware"],
            )
        return metadata
