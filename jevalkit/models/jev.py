"""TypeSafe's native System One request and image-asset transport."""

import base64
import copy
import uuid

from ._http import HttpDecisionModel, media_data_url
from ..schema import Unsupported, is_message_state


class JevModel(HttpDecisionModel):
    retryable_statuses = HttpDecisionModel.retryable_statuses | {529}

    def __init__(self, *, images=True, long_context=False, expected_response_model=None, **options):
        super().__init__(**options)
        if type(images) is not bool or type(long_context) is not bool:
            raise ValueError("Jev images and long_context options must be booleans")
        self.images, self.long_context = images, long_context
        self.expected_response_model = expected_response_model or self.model

    def request_headers(self):
        headers = super().request_headers()
        if self.images:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        if self.long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return headers

    def build_payload(self, request, *, media_root=None):
        state, questions = copy.deepcopy(request["state"]), copy.deepcopy(request["questions"])
        for question in questions.values():
            if question["type"] == "score" and len(question["criteria"]) > 10:
                raise Unsupported("Jev supports at most 10 score levels; all levels were retained")
            if question["type"] == "choice" and len(question["criteria"]) > 255:
                raise Unsupported("Jev supports at most 255 choice candidates; all candidates were retained")
        assets = {}
        if is_message_state(state):
            for message in state["messages"]:
                if message["role"] != "user":
                    raise Unsupported("Jev decision input supports user messages only")
                if isinstance(message["content"], str):
                    continue
                content = []
                for part in message["content"]:
                    kind = part["type"]
                    if kind == "text":
                        content.append(part["text"])
                    elif kind == "image_url":
                        if not self.images:
                            raise Unsupported("Jev image assets require images=true")
                        url = part["image_url"]["url"]
                        if url.startswith(("http://", "https://")):
                            raise Unsupported("Materialize Jev images locally or use image data URIs")
                        encoded = media_data_url(url, media_root)
                        if not encoded.startswith("data:") or ";base64," not in encoded:
                            raise Unsupported("Jev images require base64 image data")
                        data = base64.b64decode(encoded.split(",", 1)[1], validate=True)
                        if data.startswith(b"\x89PNG\r\n\x1a\n"):
                            image_format = "png"
                        elif data.startswith(b"\xff\xd8\xff"):
                            image_format = "jpeg"
                        else:
                            raise Unsupported("Jev assets support PNG and JPEG images; bytes were not converted")
                        placeholder = str(uuid.uuid4())
                        assets[placeholder] = {"type": "image", "format": image_format,
                                               "data_base64": base64.b64encode(data).decode("ascii")}
                        content.append(placeholder)
                    elif kind == "video_url":
                        raise Unsupported("Jev has no native video input")
                    else:
                        raise Unsupported("Unsupported Jev evidence part: " + kind)
                message["content"] = content
        payload = {"model": self.model, "state": state, "questions": questions}
        if assets:
            payload["assets"] = assets
        return payload

    def predict(self, request, *, media_root=None, assets=None):
        response = self.post(self.build_payload(request, media_root=media_root))
        if response.get("model") != self.expected_response_model:
            raise ValueError("Jev response model does not match expected_response_model")
        return response

    def metadata(self):
        return {**super().metadata(), "media_format": "assets",
                "expected_response_model": self.expected_response_model,
                "image_transport": "typesafe-assets-v1", "images": self.images,
                "long_context": self.long_context, "max_score_levels": 10, "max_choices": 255,
                "hardware": "TypeSafe hosted API (undisclosed)"}
