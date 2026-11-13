# Model integrations

**English** · [简体中文](models_zh-CN.md)

[Project overview](../README.md) · [Running evaluations](quickstart.md) · [Scoring protocols](benchmarks.md)

Twenty-five families are registered, plus the Uniform and general LLM baselines; size variants do not increase that count.
Native adapters invoke the author release. HTTP adapters require the author server or hosted API.
Registration is not evidence of a successful weight-level evaluation: consult the leaderboard and its manifests.

| Name | Family | Runtime | Setup | Source |
|---|---|---|---|---|
| Valen-0.8B | Valen | native | jevalkit[hf] | [source](https://github.com/Valen-Team/Valen) |
| Intern-Decision-0.8B | Intern-Decision | native | Python 3.12+; author requirements.txt | [source](https://github.com/InternLM/Intern-Decision) |
| Valen-2B | Valen | native | jevalkit[hf] | [source](https://github.com/Valen-Team/Valen) |
| Intern-Decision-2B | Intern-Decision | native | Python 3.12+; author requirements.txt | [source](https://github.com/InternLM/Intern-Decision) |
| Valen-4B | Valen | native | jevalkit[hf] | [source](https://github.com/Valen-Team/Valen) |
| Intern-Decision-4B | Intern-Decision | native | Python 3.12+; author requirements.txt | [source](https://github.com/InternLM/Intern-Decision) |
| Valen-Decision-1.5-{0.8B,2B,4B,9B,27B} | Valen | native | jevalkit[hf] | [source](https://huggingface.co/Valen-Team) |
| Laya | Laya | native | laya | [source](https://github.com/NandhaKishorM/laya) |
| Liquid-d1-3B | Liquid-d1 | native | transformers>=5.14; author dependencies | [source](https://huggingface.co/LiquidAI/d1-3B) |
| Decision1-Kai | vLLM-Decision1 | native | jevalkit[hf] | [source](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B) |
| Clef | Clef | native | author requirements.txt | [source](https://huggingface.co/Cloudflare/clef) |
| Clef-Flash | Clef | native | author requirements.txt | [source](https://huggingface.co/Cloudflare/clef-flash) |
| Pplx-Decider | Pplx-Decider | native | Python 3.12+; author requirements.txt | [source](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| JevK5 | JevK5 | native | jevk5 | [source](https://huggingface.co/alibiserikbay/JevK5) |
| StartLux-0.8B | StartLux | native | author requirements.txt | [source](https://huggingface.co/startlux-models/StartLux-Decision-v1-0.8B) |
| StartLux-v1.1-{0.8B,2B,4B,9B,27B}-Instant | StartLux | http | startlux-decide; STARTLUX_ENDPOINT | [source](https://github.com/StartLuxLabs/StartLux-Decision) |
| Crivo-150M | Crivo | native | jevalkit[hf,decision-index] | [source](https://huggingface.co/tardellirs/crivo-150m) |
| RSI-Jev-v4 | RSI-Jev | native | author rsijev package; `--option source=<checkout>` | [source](https://github.com/Shanghua-Gao/RSI-Jev) |
| Vjev-4B | Vjev | native | author requirements.txt | [source](https://huggingface.co/yah01/vjev-vision) |
| QEV | QEV | native | author veyra package; `--option source=<checkout>/src` | [source](https://github.com/ken-jo/qev) |
| OmniJev | OmniJev | native | author mso package; `--option source=<checkout>` `--option base_model=<Qwen3.5 weights>` | [source](https://github.com/tinnel123666888/OmniJev) |
| Visual-Jev | Visual-Jev | native | author vdm package; `--option source=<checkout>/code` `--option base_model=<Qwen3-VL weights>` | [source](https://github.com/guanxuyu-sv/Visual-Jev) |
| Amos-1-Large | Amos | native | Python 3.12+; transformers 5.17; author serve/pyproject.toml | [source](https://huggingface.co/mukaddamzaid/amos-1-large) |
| vLLM-SR-d3-turbo | vLLM-SR-d3 | native | Python 3.12+; transformers 5.17; author requirements.txt | [source](https://huggingface.co/vllm-sr/d3-turbo) |
| WeMM-JEV | WeMM-JEV | native | Python 3.12+; transformers 5.17; author requirements.txt | [source](https://huggingface.co/Rover00/WeMM-JEV) |
| Xing-VL2.0-Decision | Xing-VL | native | Python 3.12+; transformers 5.17; author requirements.txt | [source](https://huggingface.co/RBDash-Team/Xing-VL2.0-Decision) |
| Jev | Jev | http | Jev API access and JEV_ENDPOINT (full /v1/systemone URL) | [source](https://typesafe.ai) |
| OpenAI-Decisions | OpenAI-Decisions | http | OPENAI_API_KEY | [source](https://developers.openai.com/api/docs/guides/decisions) |
| LLM-OpenAI | LLM | http | OPENAI_API_KEY; `--option model=<model ID>` (default gpt-4o-mini) | [source](https://platform.openai.com/docs/guides/structured-outputs) |
| LLM-Anthropic | LLM | http | ANTHROPIC_API_KEY; `--option model=<model ID>` (default claude-haiku-4-5) | [source](https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs) |
| LLM-Gemini | LLM | http | GEMINI_API_KEY; `--option model=<model ID>` (default gemini-3.5-flash-lite) | [source](https://ai.google.dev/gemini-api/docs/structured-output) |
| Kev | Kev | http | author kev.serve instance; KEV_ENDPOINT (full /v1/systemone URL) | [source](https://github.com/jaredpalmer/kev) |
| H2O-Lightning | H2O-Lightning | http | author serve.sh instance; H2O_ENDPOINT (full /v1/systemone URL) | [source](https://huggingface.co/h2oai/h2o-lightning-4b) |
| OpenJev | OpenJev | http | author helper/shim.py instance; OPENJEV_ENDPOINT (full /v1/systemone URL) | [source](https://huggingface.co/openjev/openjev) |
| KnowLine-{0.8B-Gen1,2B-Gen1,4B-Gen5.5,9B-Gen1,26B-A4B-Gen2.5,31B-Gen1.5,35B-A3B-Gen1.5} | KnowLine | http | the model repo's serve_knowline.sh instance; KNOWLINE_ENDPOINT (full /v1/systemone URL) | [source](https://huggingface.co/PelaAI) |

Model overrides use `--option model=/path/to/checkpoint`; external author packages can use
`--option source=/path/to/checkout`. OmniJev and Visual Jev additionally require `--option base_model=/path/to/base`.

Valen supports native text, image and video. Intern-Decision, RSI-Jev, Vjev, QEV, Liquid d1
and StartLux use the authors' text/image call conventions. Visual Jev requires one image; OmniJev
requires visual evidence. Crivo’s released Decision Index engine supports text Choice/Noul.
Text-only or API video limitations are explicit errors, not silent evidence removal.

Intern-Decision's released symbol table limits it to 62 candidates; QEV's SDK permits
16 candidates and one image. Inputs beyond these limits are recorded as unsupported,
with every original option retained in the saved benchmark. Clef uses the native
encoder and head with a complete-input length check instead of the convenience
wrapper's silent state slicing. StartLux v1 can explicitly enable the author's slow
Torch path with `--option allow_slow=true --option graphs=false`; metadata records
kernel availability, graph use and calibration.
Decision1-Kai and StartLux allow at most ten Score levels. Vjev and OmniJev have
instruction-only Noul heads, so custom true/false rubrics are retained in their
instructions. Text-only SDK fields receive structured descriptions as compact JSON;
an omitted Choice description uses its candidate ID. Known legacy input-contract
failures remain in detailed reports and are excluded from the overview.

OmniJev's extra abstention probability is retained in `native_answers`. Its Choice
probabilities for closed-set benchmark scoring are explicitly conditioned on the
supplied candidates, after validating the full native distribution. A completely
abstained answer is recorded as a refusal. This policy is recorded in run metadata.

Kev, H2O-Lightning and OpenJev use their published Jev-compatible server contracts.
Set KEV_ENDPOINT, H2O_ENDPOINT or OPENJEV_ENDPOINT to a complete `/v1/systemone` URL.
Jev uses JEV_ENDPOINT and TYPESAFE_API_KEY (or `--option token_env=JEV_API_KEY`).
The registered release is `jev-1.14.0`; each response is checked against that version.
Images use native TypeSafe assets with the required image header.
Install `jevalkit[api]` and set `--option prepare_images=true` to apply the author's
image preparation: upright PNGs, aspect-preserving resizing to 280–1568 pixels per side,
without cropping or padding. Unsupported aspect ratios remain explicit.
`--option long_context=true` enables the author's 262k context mode; both feature flags are recorded.
OpenAI uses OPENAI_API_KEY.
Tokens are read from environment variables and never included in result manifests.
Once these variables are configured, the [hosted API matrix](../examples/hosted-apis.json)
runs full JevBench for both services with `python -m jevalkit matrix examples/hosted-apis.json`.
This matrix has no stored credentials. Live API evaluation requires credentials for each service.
OpenJev's single screenshot is sent through the author's `state.image` field;
H2O-Lightning uses its top-level `images` field. Protocol/localhost fixture tests
are distinct from actual server-weight evaluations and never create leaderboard scores.

H2O-Lightning ships `model.safetensors` for the backbone and
`label_rows.safetensors` for the author's separate decision readout. Before
starting its published `serve.sh`, create a standard HF index that selects the
backbone file for vLLM's local loader:

```bash
python -m jevalkit.models.h2o_lightning /path/to/h2o-lightning-4b model.safetensors
bash /path/to/h2o-lightning-4b/serve.sh
```

The shim reads the original label rows separately and applies them to the pooled
hidden state. The index records all backbone tensor names; its preparation
rejects conflicting existing selections. The deployment fingerprint includes
both original weight files and the generated index.

For OpenJev, use the pinned release's recommended `READOUT_TARGETED=1` server
path to retrieve every candidate token's score. Its old default path fills
candidates missing from the vocabulary's top-K with a fixed log probability.
The tested reference deployment uses the author's published `READOUT_T=0.85`,
`READOUT_NOUL_T=1.829074`, `READOUT_NOUL_BIAS=0`, `READOUT_INSTR_STYLE=pyrepr`
and FP8 serving configuration. These constants are not fitted on JevalKit's
test sets; the service provenance records all selected settings.

Valen-Decision-1.5 uses the Valen adapter. Its leaderboard runs used
`--option attention=flash_attention_2`; Decision Index vision applies the same
1,638,400-pixel Valen image budget.

Amos-1-Large, vLLM-SR-d3-turbo, WeMM-JEV and Xing-VL2.0-Decision run the inference
code shipped inside each checkpoint: `serve/amos`, `d3_engine.py`, `inference/` and
`rbdash_decision.py`. A same-named module imported from elsewhere is rejected. All
four take text and images; video is reported as unsupported. Amos follows the
author's serving defaults (`max_len` 131,072, one rotation, strict inputs) and records
an out-of-GPU-memory request as unsupported with its own message. d3 and WeMM-JEV
accept 2 to 10 Score levels. Xing scores each question in a separate forward pass and
accepts at most 4 images and 255 options.

Intern-Decision accepts at most 16 questions in a shared request. Decision1-Kai
requires nonempty state text. Requests outside these released SDK limits are
recorded as unsupported with the original request preserved.

StartLux v1.1 Instant uses the author’s `startlux-decide serve` with `STARTLUX_ENDPOINT`
(the full `/v1/systemone` URL) and optional `STARTLUX_API_KEY`. The service must load
the selected checkpoint; a custom service name requires `--option expected_response_model=NAME`.
Requests fix `think_mode=never`, `allow_none=false` and `none_check=true`, recorded in the manifest.
Use the [managed-service example](../examples/served-model.json) to record weights and author source.

## Tested runtimes

| Runtime | Python | PyTorch / CUDA | Transformers |
|---|---|---|---|
| Valen | 3.10 | 2.6.0 / 12.4 | 5.4.0 |
| Native author SDKs, Valen-Decision-1.5 | 3.12 | 2.10.0 / 12.8 | 5.14.1 |
| Amos / d3 / WeMM-JEV / Xing-VL checkpoint engines | 3.12 | 2.10.0 / 12.8 | 5.17.0 |
| H2O-Lightning / OpenJev services | Separate vLLM 0.30.0 environment | 2.13.0 / 13.0 | 5.19.0 |

Use separate environments where author dependencies differ. Matrix rows can
select an absolute `python` path; SDK source checkouts use `--option source=PATH`.
Each result manifest records the actual library versions, weights and native
source fingerprints. See [running evaluations](quickstart.md) for usage.

## Add a model

Create `jevalkit/models/<model>.py`, implement the interface in `base.py`, and
register its import path in `jevalkit/config.py`. Size variants can share the same
implementation. Keep author imports inside initialization so other models remain
usable without that SDK. Reuse `_http.py` or `_hf.py` for matching calling conventions.
Run `jeval check --model <name> --output check.json` and the relevant adapter tests.

## General LLM baselines

`LLM-OpenAI`, `LLM-Anthropic` and `LLM-Gemini` run general-purpose LLMs on the same benchmarks,
so decision models can be compared with them on agreement, latency and token usage. Each request
sends the evidence (text and images) and every question with its candidates, and asks for a
probability per candidate through the provider's JSON-schema structured output. JevalKit
normalizes each question's probabilities and records token usage in `predictions.jsonl`.
`LLM-OpenAI` uses Chat Completions, so `--option endpoint=<URL>` also reaches OpenAI-compatible
providers such as Fireworks, Groq or a local vLLM server; pass `--option token_env=<VARIABLE>` for
their key. Video inputs are reported as unsupported; extract frames as images instead.

```bash
python -m jevalkit run --model LLM-OpenAI --option model=gpt-4o-mini --data VisualDecisionBench_Image \
  --output "$JEVALKIT_DATA_ROOT/results"
```
