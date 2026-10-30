# Workflows, cost and weak slices

Benchmarks score single questions. A workflow scores what an application does with the answers: the model answers typed questions about each case, fixed code turns the answers into actions, and the actions are compared with reference decisions. The workflow code is the same for every model, so differences in agreement come from the answers.

## Run a workflow

```bash
pip install -e ".[workflow]"
jeval workflow list
jeval workflow run delivery_support --model Valen-API-4B --workers 4 --output runs/delivery_support/valen-4b
jeval workflow run invoice_processing --model Valen-4B --output runs/invoice_processing/valen-4b
jeval workflow compare invoice_processing runs/invoice_processing/*
```

Any model in `jeval list models` works, including `LLM-*` baselines, which answer the same questions through structured output. `--workers` runs cases concurrently; keep 1 for a local GPU model. `--limit N` runs the first N cases as a smoke test, and `--resume` continues an interrupted run with the same code, data, model and options.

| Workflow | Cases | Policies | Reference | Source |
| --- | ---: | --- | --- | --- |
| `invoice_processing` | 150 | `startup`, `enterprise`, `high_volume_retailer` | OpenAI, Anthropic and consensus model runs | TypeSafe's [WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) code and [dataset](https://huggingface.co/datasets/typesafeAI/evalsafe-invoice-processing), Apache-2.0 |
| `delivery_support` | 60 | `fastbite` | Human labels | The [support-router recipe](https://valen-team.github.io/cookbook/skill-support-router/) on the hidden test set of the [skill benchmark](https://github.com/Valen-Team/skills/tree/main/evals) |

Each invoice case sends about 46 questions over a 20–35k character packet in one request. The free hosted API takes 32,000 characters of state and 300 characters per criterion (the adapter splits requests into groups of 8 questions), so run this workflow on a local or keyed deployment; requests over those limits are recorded as `unsupported` and count as disagreement. `delivery_support` fits the free API.

The invoice port is checked by replaying the dataset's consensus answers through the vendored code, which reproduces every consensus decision (`tests/test_workflows.py`). The delivery-support thresholds were tuned on `valen-4b`, so they favour that size.

## Scores

A run folder holds `manifest.json`, the `cases.jsonl` journal and `scores.json`. Each case records its decisions, every model call with its answers, latency, token counts and estimated cost.

- `exact_actions`: the action set, including arguments, equals the reference set; order and duplicates are ignored.
- `primary_action`: the primary action equals the reference's.

Both are averaged over policies. Cases that fail or exceed a model's limits remain in the denominator. `jeval workflow compare` lists local runs next to TypeSafe's published invoice runs (Jev 1.13 and eight LLMs) with cost and time per case. The published models also produced the reference labels, so they agree with it more easily than an independent model does.

## Cost

`jevalkit/pricing.json` lists USD per million input and output tokens by `provider:model`, from WorkflowEvals' published rates. Benchmark and workflow reports add token totals and estimated cost when the rate and usage are known; otherwise cost is `null`. For an LLM served through an OpenAI-compatible gateway, name the lab whose list price applies:

```bash
jeval run --model LLM-OpenAI --data JevBench --output runs \
  --option endpoint=https://gateway.example/v1/chat/completions --option token_env=GATEWAY_API_KEY \
  --option model=claude-sonnet-5 --option billing=anthropic
```

Self-hosted Valen has no token price; compare it by time per case, or add a rate to `pricing.json`.

## Find weak slices

Benchmark reports group every metric by question type, subset, domain, modality and three properties of the input: `script` (`cjk`, `latin`, `other`), `state_size` (`<1k`, `1k-4k`, `4k-16k`, `16k+` characters) and `candidates` (`2`, `3-4`, `5-8`, `9+`). Re-score older runs with `jeval score RUN` to add them.

```bash
jeval analyze --targets runs/Valen-*/JevBench --references runs/Intern-Decision-*/JevBench runs/StartLux-*/JevBench \
  --questions --output analysis
```

A slice is flagged when it has at least `--min-questions` hard questions and every target's accuracy is at least `--gap` (default 0.05) below both its own overall accuracy and the median of the reference group. `--questions` also lists hard questions every target answers wrong while at least `--min-reference-share` (default 0.75) of the references answer right; it needs the benchmark data under `--root`. A flagged slice is where to look, not a conclusion: read its questions, then confirm with targeted variations before recording a limitation.
