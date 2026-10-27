<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/jevalkit-dark.svg">
    <img src="assets/branding/jevalkit-light.svg" alt="JevalKit evaluation toolkit for decision models" width="680">
  </picture>
</h1>

<p align="center">
  <b>A toolkit for evaluating Jev-like decision models.</b>
</p>

<p align="center">
  <a href="#supported-models"><img src="https://img.shields.io/badge/Models-20_families-007F6B?style=flat&amp;labelColor=1E232C" alt="20 model families"></a>
  <a href="#supported-benchmarks"><img src="https://img.shields.io/badge/Benchmarks-8_suites-007F6B?style=flat&amp;labelColor=1E232C" alt="8 Jev-related benchmark suites"></a>
  <a href="https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard"><img src="https://img.shields.io/badge/Space-Leaderboard-007F6B?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=1E232C" alt="Hugging Face Space: leaderboard"></a>
  <a href="https://huggingface.co/datasets/Valen-Team/JevalKit_Datasets"><img src="https://img.shields.io/badge/Datasets-10_TSV_views-007F6B?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=1E232C" alt="Hugging Face: 10 dataset views"></a>
  <a href="https://huggingface.co/Valen-Team"><img src="https://img.shields.io/badge/Hugging_Face-Valen_Team-007F6B?style=flat&amp;logo=huggingface&amp;logoColor=white&amp;labelColor=1E232C" alt="Hugging Face: Valen Team"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-555555?style=flat&amp;labelColor=1E232C" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-555555?style=flat&amp;logo=python&amp;logoColor=FFD43B&amp;labelColor=1E232C" alt="Python 3.10+"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh-CN.md">简体中文</a><br>
  <a href="https://valen-team.github.io/docs/jevalkit/">Documentation</a> · <a href="https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard">Leaderboard</a> · <a href="https://huggingface.co/datasets/Valen-Team/JevalKit_Datasets">Datasets</a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> · <a href="#latest-updates">News</a> · <a href="#quick-start">Quick start</a> · <a href="#supported-models">Models</a> · <a href="#supported-benchmarks">Benchmarks</a> · <a href="#evaluation-results">Results</a> · <a href="#documentation">Local guides</a>
</p>

<a id="introduction"></a>

## Introduction

JevalKit is an open-source evaluation toolkit for Jev-like decision models,
providing unified model interfaces and prepared evaluation datasets. Users can
evaluate models across benchmarks with a single command, without preparing data
in separate repositories.

**Integration requests:** Simply [open an issue](https://github.com/Valen-Team/JevalKit/issues)
with the model or benchmark ID, and we can help add support for it.

<a id="highlights"></a>
<a id="latest-updates"></a>

## News

- **2026-10-19:** Released the [🤗 JevalKit Leaderboard](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard)
  with benchmark and model-size filters, model comparison, probability metrics
  and Markdown, TSV and JSON result exports. Complete evaluations, smoke tests
  and imported predictions are distinguished.
- **2026-10-18:** Integrated eight benchmarks: VisualDecisionBench, JevBench,
  Decision Index (public components), Typed-Decisions, DecisionBench, Procedural
  Typed Decisions, zh-decision-bench and JevAlt-Bench. Prepared evaluation datasets
  are available in [🤗 JevalKit Datasets](https://huggingface.co/datasets/Valen-Team/JevalKit_Datasets).
- **2026-10-18:** Added support for 20 Jev-like model families, with real inference
  verified for 18. Hosted API adapters for Jev and OpenAI Decisions are implemented;
  live evaluations remain pending.

<a id="quick-start"></a>

## Quick Start

### 1. Install

```bash
git clone https://github.com/Valen-Team/JevalKit.git
cd JevalKit
python -m pip install -e '.[hf,data]'
```

The CLI and scorers require Python 3.10+. Model inference uses the dependencies
specified by each author; some models require Python 3.12+ or a separate serving
environment. See the [model integration guide](docs/models.md).

### 2. Download a model and dataset

```bash
export JEVALKIT_DATA_ROOT="$PWD/data"

hf download Valen-Team/Valen-0.8B \
  --local-dir "$JEVALKIT_DATA_ROOT/models/Valen-0.8B"

hf download Valen-Team/JevalKit_Datasets --repo-type dataset \
  --include "datasets/JevBench.tsv" --local-dir "$JEVALKIT_DATA_ROOT"
```

The published TSVs are ready to evaluate. To download all published views, replace
`datasets/JevBench.tsv` with `datasets/*.tsv`. Visual media and Decision Index
scoring inputs have separate downloads in the [usage guide](docs/quickstart.md#download-data).

### 3. Run and compare

```bash
python -m jevalkit run --model Valen-0.8B --data JevBench \
  --option model="$JEVALKIT_DATA_ROOT/models/Valen-0.8B" \
  --output "$JEVALKIT_DATA_ROOT/results"

python -m jevalkit leaderboard "$JEVALKIT_DATA_ROOT/results" \
  --output "$JEVALKIT_DATA_ROOT/leaderboard"
```

Pass `--option model=/path/to/checkpoint` to reuse existing weights. Use
`--limit 10` for a smoke test, or `--resume` to continue an interrupted run with
the same settings. [Full usage guide →](docs/quickstart.md)

<a id="supported-models"></a>

## Supported Models

Size variants and Clef/Clef-Flash count within their respective families.
**Measured** means real inference has been verified for at least one registered
variant; per-benchmark coverage is shown in the [leaderboard](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard).

<details>
<summary>View supported models (20 families)</summary>

| Model family | Registered model IDs | Interface | Evaluation |
|---|---|---|---|
| [Valen](https://github.com/Valen-Team/Valen) | `Valen-0.8B`, `Valen-2B`, `Valen-4B` | Native | Measured |
| [Intern-Decision](https://github.com/InternLM/Intern-Decision) | `Intern-Decision-0.8B`, `Intern-Decision-2B`, `Intern-Decision-4B` | Native | Measured |
| [Laya](https://github.com/NandhaKishorM/laya) | `Laya` | Native | Measured |
| [Liquid d1](https://huggingface.co/LiquidAI/d1-3B) | `Liquid-d1-3B` | Native | Measured |
| [vLLM Decision1](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B) | `Decision1-Kai` | Native | Measured |
| [Clef](https://huggingface.co/Cloudflare/clef) | `Clef`, `Clef-Flash` | Native | Measured |
| [Pplx-Decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | `Pplx-Decider` | Native | Measured |
| [JevK5](https://huggingface.co/alibiserikbay/JevK5) | `JevK5` | Native | Measured |
| [StartLux](https://github.com/StartLuxLabs/StartLux-Decision) | `StartLux-0.8B`, `StartLux-v1.1-{0.8B,2B,4B,9B,27B}-Instant` | Native / Author server | v1 measured; v1.1 pending |
| [Crivo](https://huggingface.co/tardellirs/crivo-150m) | `Crivo-150M` | Native | Measured |
| [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) | `RSI-Jev-v4` | Native | Measured |
| [Vjev](https://huggingface.co/yah01/vjev-vision) | `Vjev-4B` | Native | Measured |
| [QEV](https://github.com/ken-jo/qev) | `QEV` | Native | Measured |
| [OmniJev](https://github.com/tinnel123666888/OmniJev) | `OmniJev` | Native | Measured |
| [Visual-Jev](https://github.com/guanxuyu-sv/Visual-Jev) | `Visual-Jev` | Native | Measured |
| [Kev](https://github.com/jaredpalmer/kev) | `Kev` | Author server | Measured |
| [H2O-Lightning](https://huggingface.co/h2oai/h2o-lightning-4b) | `H2O-Lightning` | Author server | Measured |
| [OpenJev](https://huggingface.co/openjev/openjev) | `OpenJev` | Author server | Measured |
| [Jev](https://typesafe.ai) | `Jev` | Hosted API | Full evaluation available |
| [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) | `OpenAI-Decisions` | Hosted API | Live evaluation pending |

</details>

The catalog also includes a `Uniform` diagnostic baseline. Author dependencies,
context limits, supported question types and media contracts are documented in
the [integration guide](docs/models.md).

<a id="supported-benchmarks"></a>

## Supported Benchmarks

<details>
<summary>View supported datasets (8 benchmark suites)</summary>

| Benchmark | Input | Evaluation data | Main measurements |
|---|---|---|---|
| [VisualDecisionBench](https://huggingface.co/datasets/Valen-Team/VisualDecisionBench) | 🖼️ Image · 🎬 Video | 1,696 states; 9,143 questions | Accuracy, soft-target Brier/NLL, ordinal metrics |
| [JevBench](https://github.com/fstandhartinger/jevbench) | 📝 Text | 231 public tasks across easy/original/hard tiers | Official task scores and calibration |
| [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) | 📝 Text · 🖼️ Image | 0.4.1 public tracks: 140,178 text requests; 12,210 vision questions | Official weighted public text and vision indices |
| [Typed-Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 📝 Text | 400 states; 2,000 decisions | Modal accuracy, KL and soft-target Brier |
| [DecisionBench](https://huggingface.co/datasets/Hanno-Labs/decision-bench) | 📝 Text | 23,900 evaluation rows; 43 tasks | Accuracy, reference NLL and calibration |
| [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | 📝 Text | 18,000 test states; 57,016 questions | Rule execution and exact probability judgments |
| [zh-decision-bench](https://github.com/CodyQin/zh-decision-bench) | 📝 Chinese | v0.2: 378 items; 467 questions | Routing, moderation, ordinal ratings and calibration |
| [JevAlt-Bench](https://huggingface.co/datasets/mertkayacs/jevalt-bench) | 📝 EN · TR · DE | 3,116 test states; 8,434 questions | Labeled and soft-probability decisions; language slices |

</details>

VisualDecisionBench's image and video views belong to the same benchmark.
Decision Index public scores and its private leaderboard components are reported
separately. See the [dataset packages](jevalkit/datasets/) and
[scoring protocols](docs/benchmarks.md).

Download the prepared TSVs from [🤗 JevalKit Datasets](https://huggingface.co/datasets/Valen-Team/JevalKit_Datasets).
Use the [matrix example](examples/decision-benchmarks.json) to evaluate several
models and benchmarks.

<a id="evaluation-results"></a>

## Evaluation Results

<!-- jevalkit:results-summary:start -->
The results summary recorded on 2026-10-27 contains **214 eligible full scores** and **300 retained
run artifacts**, covering **19 measured model families**; OpenAI-Decisions
remains pending. The leaderboard is a separately published snapshot and may differ. [🤗 Open the leaderboard](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard).
<!-- jevalkit:results-summary:end -->

<a id="documentation"></a>

## Documentation

Read the [JevalKit documentation](https://valen-team.github.io/docs/jevalkit/) on the website, or use the repository guides below.

| Guide | Contents |
|---|---|
| [Running evaluations](docs/quickstart.md) | Downloads, evaluation commands, resume, APIs and parallel runs |
| [Model integrations](docs/models.md) | Author SDKs, runtime dependencies and input limits |
| [Scoring protocols](docs/benchmarks.md) | Dataset definitions, probability metrics and official scorers |

<a id="contributing"></a>

## Contributing

Each model implementation has its own file in `jevalkit/models/`, registered in
[`jevalkit/config.py`](jevalkit/config.py). Each benchmark has its own directory
in `jevalkit/datasets/`, containing preparation, scoring and frozen TSV metadata.
Pull requests are welcome from anyone interested in contributing to the project.

Bug reports and contributions are welcome through
[Issues](https://github.com/Valen-Team/JevalKit/issues) and
[Pull Requests](https://github.com/Valen-Team/JevalKit/pulls).

### Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Liuziyu77"><img src="https://github.com/Liuziyu77.png?size=128" width="64" height="64" alt="Liuziyu77"><br><sub><b>Liuziyu77</b></sub></a>
    </td>
    <td align="center">
      <a href="https://github.com/yph22"><img src="https://github.com/yph22.png?size=128" width="64" height="64" alt="yph22"><br><sub><b>yph22</b></sub></a>
    </td>
    <td align="center">
      <a href="https://github.com/SYuan03"><img src="https://github.com/SYuan03.png?size=128" width="64" height="64" alt="SYuan03"><br><sub><b>SYuan03</b></sub></a>
    </td>
    <td align="center">
      <a href="https://github.com/yuhangzang"><img src="https://github.com/yuhangzang.png?size=128" width="64" height="64" alt="yuhangzang"><br><sub><b>yuhangzang</b></sub></a>
    </td>
  </tr>
</table>

<a id="license-and-acknowledgements"></a>

## License and Acknowledgements

JevalKit is released under [Apache-2.0](LICENSE).
We thank [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for its evaluation
toolkit design, and the authors of the integrated decision models and benchmarks.


## Probability reliability and automation

Reuse per-question predictions offline, fit and save temperatures, freeze automation thresholds on separate validation data, then measure held-out coverage and accepted error. Reports include before/after NLL, Brier, ECE, reliability curves and failure counts, without changing official scoring. See the [workflow and cache protocol](docs/reliability.md). Fitting and replay need no model downloads.
