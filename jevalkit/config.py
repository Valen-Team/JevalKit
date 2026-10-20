"""Explicit model-family registry. Sizes are variants, not additional families."""

import importlib
import os
from dataclasses import asdict, dataclass, field

from .security import is_credential_key


@dataclass(frozen=True)
class ModelSpec:
    family: str
    adapter: str
    model: str
    source_url: str
    options: dict = field(default_factory=dict)
    transport: str = "native"
    requirements: str = "jevalkit[hf]"


MODELS = {}


def register(name, family, adapter, model, source_url, **kwargs):
    MODELS[name] = ModelSpec(family, adapter, model, source_url, **kwargs)


for size in ("0.8B", "2B", "4B"):
    register(
        "Valen-" + size,
        "Valen",
        "jevalkit.models.valen:ValenModel",
        "Valen-Team/Valen-" + size,
        "https://github.com/Liuziyu77/Valen",
    )
    register(
        "Intern-Decision-" + size,
        "Intern-Decision",
        "jevalkit.models.intern_decision:InternDecision",
        "internlm/Intern-Decision-" + size,
        "https://github.com/InternLM/Intern-Decision",
        requirements="Python 3.12+; author requirements.txt",
    )
register(
    "Laya",
    "Laya",
    "jevalkit.models.laya:LayaModel",
    "convaiinnovations/laya",
    "https://github.com/NandhaKishorM/laya",
    requirements="laya",
)
register(
    "Liquid-d1-3B",
    "Liquid-d1",
    "jevalkit.models.liquid:LiquidModel",
    "LiquidAI/d1-3B",
    "https://huggingface.co/LiquidAI/d1-3B",
    requirements="transformers>=5.14; author dependencies",
)
register(
    "Decision1-Kai",
    "vLLM-Decision1",
    "jevalkit.models.decision1:Decision1Model",
    "vllm-sr/Decision-1.0-Kai-0.6B",
    "https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B",
    options={"image_support": False, "max_score_levels": 10, "requires_nonempty_state": True},
)
register(
    "Clef",
    "Clef",
    "jevalkit.models.clef:ClefModel",
    "Cloudflare/clef",
    "https://huggingface.co/Cloudflare/clef",
    requirements="author requirements.txt",
)
register(
    "Clef-Flash",
    "Clef",
    "jevalkit.models.clef:ClefModel",
    "Cloudflare/clef-flash",
    "https://huggingface.co/Cloudflare/clef-flash",
    requirements="author requirements.txt",
)
register(
    "Pplx-Decider",
    "Pplx-Decider",
    "jevalkit.models.pplx_decider:PplxDecider",
    "perplexity-ai/pplx-decider-v1.1-27b",
    "https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b",
    requirements="Python 3.12+; author requirements.txt",
)
register(
    "JevK5",
    "JevK5",
    "jevalkit.models.jevk5:JevK5Model",
    "alibiserikbay/JevK5",
    "https://huggingface.co/alibiserikbay/JevK5",
    requirements="jevk5",
)
register(
    "StartLux-0.8B",
    "StartLux",
    "jevalkit.models.startlux:StartLuxModel",
    "startlux-models/StartLux-Decision-v1-0.8B",
    "https://huggingface.co/startlux-models/StartLux-Decision-v1-0.8B",
    requirements="author requirements.txt",
)
for size in ("0.8B", "2B", "4B", "9B", "27B"):
    checkpoint = "startlux-models/StartLux-Decision-v1.1-" + size + "-Instant"
    register(
        "StartLux-v1.1-" + size + "-Instant",
        "StartLux",
        "jevalkit.models.startlux:StartLuxInstant",
        checkpoint,
        "https://huggingface.co/" + checkpoint,
        options={"endpoint_env": "STARTLUX_ENDPOINT", "token_env": "STARTLUX_API_KEY"},
        transport="http",
        requirements="author startlux-decide server; STARTLUX_ENDPOINT (full /v1/systemone URL)",
    )
register(
    "Crivo-150M",
    "Crivo",
    "jevalkit.models.crivo:CrivoModel",
    "tardellirs/crivo-150m",
    "https://huggingface.co/tardellirs/crivo-150m",
    requirements="jevalkit[hf,decision-index]",
)
register(
    "RSI-Jev-v4",
    "RSI-Jev",
    "jevalkit.models.rsi_jev:RSIJevModel",
    "shgao/rsi-jev-v4.0-vl-qwen3.5-2b",
    "https://github.com/Shanghua-Gao/RSI-Jev",
    requirements="author rsijev package; --option source=<checkout>",
)
register(
    "Vjev-4B",
    "Vjev",
    "jevalkit.models.vjev:VjevModel",
    "yah01/vjev-vision",
    "https://huggingface.co/yah01/vjev-vision",
    requirements="author requirements.txt",
)
register(
    "QEV",
    "QEV",
    "jevalkit.models.qev:QEVModel",
    "ken-jo/qev",
    "https://github.com/ken-jo/qev",
    requirements="author veyra package; --option source=<checkout>/src",
)
register(
    "OmniJev",
    "OmniJev",
    "jevalkit.models.omnijev:OmniJevModel",
    "tinnel123/OmniJev-0.8B",
    "https://github.com/tinnel123666888/OmniJev",
    requirements="author mso package; --option source=<checkout> --option base_model=<Qwen3.5 weights>",
)
register(
    "Visual-Jev",
    "Visual-Jev",
    "jevalkit.models.visual_jev:VisualJevModel",
    "guanxuyu/visual-jev-4b-answer-sft",
    "https://github.com/guanxuyu-sv/Visual-Jev",
    requirements="author vdm package; --option source=<checkout>/code --option base_model=<Qwen3-VL weights>",
)
register(
    "Jev",
    "Jev",
    "jevalkit.models.jev:JevModel",
    "jev-1.13.0",
    "https://typesafe.ai",
    options={"endpoint_env": "JEV_ENDPOINT", "token_env": "TYPESAFE_API_KEY"},
    transport="http",
    requirements="Jev API access and JEV_ENDPOINT (full /v1/systemone URL)",
)
register(
    "OpenAI-Decisions",
    "OpenAI-Decisions",
    "jevalkit.models.openai_decisions:OpenAIDecisions",
    "gpt-6-luna",
    "https://developers.openai.com/api/docs/guides/decisions",
    transport="http",
    requirements="OPENAI_API_KEY",
)
register(
    "Kev",
    "Kev",
    "jevalkit.models.kev:KevModel",
    "jaredpalmer/kev-4b",
    "https://github.com/jaredpalmer/kev",
    options={"endpoint_env": "KEV_ENDPOINT", "token_env": "KEV_API_KEY", "media_format": "text"},
    transport="http",
    requirements="author kev.serve instance; KEV_ENDPOINT (full /v1/systemone URL)",
)
register(
    "H2O-Lightning",
    "H2O-Lightning",
    "jevalkit.models.h2o_lightning:H2OLightningModel",
    "h2oai/h2o-lightning-4b",
    "https://huggingface.co/h2oai/h2o-lightning-4b",
    options={"endpoint_env": "H2O_ENDPOINT", "token_env": "H2O_API_KEY", "media_format": "top_level"},
    transport="http",
    requirements="author serve.sh instance; H2O_ENDPOINT (full /v1/systemone URL)",
)
register(
    "OpenJev",
    "OpenJev",
    "jevalkit.models.openjev:OpenJevModel",
    "openjev/openjev",
    "https://huggingface.co/openjev/openjev",
    options={
        "endpoint_env": "OPENJEV_ENDPOINT",
        "token_env": "OPENJEV_API_KEY",
        "media_format": "state_image",
    },
    transport="http",
    requirements="author helper/shim.py instance; OPENJEV_ENDPOINT (full /v1/systemone URL)",
)
register(
    "Uniform", "baseline", "jevalkit.models.uniform:UniformModel", "uniform", "builtin", requirements="none"
)


def build_model(name, root=None, **overrides):
    from .paths import data_root

    os.environ.setdefault("HF_HOME", str(data_root(root) / "cache" / "hf"))
    os.environ.setdefault("HF_MODULES_CACHE", str(data_root(root) / "cache" / "modules"))
    spec = MODELS[name]
    options = {"model": spec.model, **spec.options, **overrides}
    endpoint_env = options.pop("endpoint_env", None)
    if endpoint_env:
        endpoint = options.get("endpoint") or os.environ.get(endpoint_env)
        if not endpoint:
            raise ValueError(
                f"{name} requires {endpoint_env} or --option endpoint=<full URL>. {spec.requirements}"
            )
        options["endpoint"] = endpoint
    if any(is_credential_key(key) for key in options):
        raise ValueError("Pass a token_env variable name, never a credential value")
    module, cls = spec.adapter.split(":")
    return getattr(importlib.import_module(module), cls)(root=root, **options)


def model_catalog():
    return {name: asdict(spec) for name, spec in MODELS.items()}
