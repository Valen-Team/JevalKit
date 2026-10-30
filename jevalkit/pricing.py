"""Estimated request cost from token usage and provider:model rates; unknown rates give unknown cost."""

import math
from pathlib import Path

from .io import load_json, object_hash

# Hosted decision APIs whose adapters do not report a provider in their runtime metadata.
ADAPTER_PROVIDERS = {"JevModel": "typesafe", "ValenAPIModel": "valen", "ValenModel": "valen",
                     "OpenAIDecisions": "openai"}


def load_pricing(path=None):
    prices = load_json(path or Path(__file__).with_name("pricing.json"))
    for key, rate in prices.items():
        if key.startswith("_"):
            continue
        if not (isinstance(rate, list) and len(rate) == 2
                and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in rate)):
            raise ValueError(f"Invalid input/output price for {key}")
    return prices


def price_key(runtime):
    provider = runtime.get("billing") or runtime.get("provider") or ADAPTER_PROVIDERS.get(runtime.get("adapter"))
    return f"{provider}:{runtime['model']}" if provider and runtime.get("model") else None


def rates_for(runtime, prices=None):
    """The [input, output] USD-per-million rate for a run, or None."""
    prices = load_pricing() if prices is None else prices
    key = price_key(runtime)
    return {"key": key, "rate": prices[key], "pricing_sha256": object_hash(prices)} if key in prices else None


def usage_tokens(response):
    usage = (response or {}).get("usage") or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


def cost_usd(rates, input_tokens, output_tokens):
    """Uncached USD, or None when the rate or billable usage is unknown."""
    if rates is None:
        return None
    rate_in, rate_out = rates["rate"]
    if input_tokens is None and rate_in == 0:
        input_tokens = 0
    if output_tokens is None and rate_out == 0:
        output_tokens = 0
    if input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * rate_in + output_tokens * rate_out) / 1e6
