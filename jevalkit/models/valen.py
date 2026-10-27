"""Native ValenModel adapter."""

import copy
import os
from .base import DecisionModel
from ._utils import render_questions, resolve_model
from .provenance import inference_artifacts, runtime_versions
from ..io import dumps, sha256
from ..paths import data_root
from ..schema import Unsupported, is_message_state


class ValenModel(DecisionModel):
    modalities = frozenset({"text", "image", "video"})

    def __init__(
        self,
        model="Valen-Team/Valen-0.8B",
        device="cuda",
        revision="main",
        attention="sdpa",
        execution="shared_state",
        max_image_pixels=None,
        root=None,
        calibration=None,
        calibration_protocol=None,
        **options,
    ):
        if max_image_pixels is not None and (type(max_image_pixels) is not int or max_image_pixels < 65536):
            raise ValueError("max_image_pixels must be an integer of at least 65536")
        os.environ.setdefault("HF_MODULES_CACHE", str(data_root(root) / "cache" / "modules"))
        from transformers import AutoModel
        import torch

        self.torch, self.device = torch, device
        self.folder = resolve_model(model, revision, root)
        self.model = (
            AutoModel.from_pretrained(
                str(self.folder),
                trust_remote_code=True,
                dtype="auto",
                attn_implementation=attention,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        self.execution = execution
        self.temperature = 1.0
        if max_image_pixels is not None:
            media = copy.deepcopy(self.model.config.valen_media_kwargs)
            size = media.setdefault("images_kwargs", {}).setdefault("size", {})
            size.update(longest_edge=max_image_pixels, shortest_edge=min(size.get("shortest_edge", 65536), max_image_pixels))
            self.model.config.valen_media_kwargs = media
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        self.info = {
            "adapter": "ValenModel",
            "device": device,
            "model": model,
            "revision": revision,
            "path": str(self.folder),
            "execution": execution,
            "attention": attention,
            "config_sha256": sha256(self.folder / "config.json"),
            "dtype": str(next(self.model.parameters()).dtype),
            "temperature": 1.0,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "versions": runtime_versions(),
            "hardware": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
            "artifacts_sha256": inference_artifacts(self.folder),
        }
        if max_image_pixels is not None:
            self.info["max_image_pixels"] = max_image_pixels
            self.info["image_budget_scope"] = "native processor images_kwargs.size.longest_edge"

        if calibration:
            from ..reliability import load_calibration, digest, read_json
            from ..reliability_export import runtime_identity
            if not calibration_protocol:
                raise ValueError("calibration_protocol JSON path is required to verify the deployment recipe")
            identity = runtime_identity(self.info, read_json(calibration_protocol))
            artifact = load_calibration(calibration, identity)
            self.temperature = artifact["temperature"]
            self.info.update(temperature=self.temperature, calibration_sha256=digest(artifact),
                             calibration_protocol_sha256=identity["protocol_sha256"])

    def predict(self, request, *, media_root=None, assets=None):
        if self.info.get("calibration_sha256") and "temperature" in request:
            raise ValueError("Request temperature cannot override a fitted calibration")
        request = copy.deepcopy(request)
        state = request["state"]
        if not isinstance(state, str) and not is_message_state(state):
            request["state"] = dumps(state)
        request["questions"] = render_questions(request["questions"])
        try:
            with self.torch.inference_mode():
                return self.model.predict(
                    request, media_root=str(media_root or "."), execution=self.execution, temperature=self.temperature
                )
        except ValueError as exc:
            if "exceed max_length" in str(exc) or "Reserved tokenizer control token" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def metadata(self):
        return self.info

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
