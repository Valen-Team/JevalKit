# Running evaluations

**English** · [简体中文](quickstart_zh-CN.md)

[Project overview](../README.md) · [Model integrations](models.md) · [Scoring protocols](benchmarks.md)

## Install

```bash
python -m pip install -e '.[hf,data]'
export JEVALKIT_DATA_ROOT="$PWD/data"
```

The CLI requires Python 3.10+. Install the selected model author's dependencies;
several native SDKs require Python 3.12+. See [tested runtimes](models.md#tested-runtimes).

## Download data

Download ready-to-use TSVs from Hugging Face:

```bash
hf download Valen-Team/JevalKit_Datasets --repo-type dataset \
  --include "datasets/*.tsv" --local-dir "$JEVALKIT_DATA_ROOT"
```

For one benchmark, replace `datasets/*.tsv` with `datasets/JevBench.tsv`.
JevalKit reads the published schema, counts and hashes from its bundled
[benchmark packages](../jevalkit/datasets/); local manifests are optional
for these published snapshots. No conversion or dataset construction is needed.

For VisualDecisionBench, download and extract the original media:

```bash
hf download Valen-Team/VisualDecisionBench assets.zip --repo-type dataset \
  --local-dir "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench"
python -m zipfile -e "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench/assets.zip" \
  "$JEVALKIT_DATA_ROOT/media/VisualDecisionBench"
```

For Decision Index, install its scorer and download the edition 0.3 public text
inputs, also used by the 0.4.1 board:

```bash
python -m pip install -e '.[decision-index]'
hf download multimodalart/decision-index-suite-0.3 \
  selected-rows.jsonl.gz added-rows.jsonl.gz gsm8k-rows.jsonl.gz \
  excluded-questions.json manifest.json --repo-type dataset \
  --local-dir "$JEVALKIT_DATA_ROOT/raw/DecisionIndex-0.3"
```

Prepare the 0.4.1 public Text/Vision TSVs with the official converters:

```bash
python -m jevalkit prepare DecisionIndex_Text \
  --source "$JEVALKIT_DATA_ROOT/raw/DecisionIndex-0.3"
python -m pip install 'pillow==12.3.0'
export DECISION_INDEX_FONT="/path/to/arialbd.ttf"
python -m jevalkit prepare DecisionIndex_Vision
```

Vision preparation downloads the pinned original sources and verifies their
reference hashes. It requires approved [Winoground access](https://huggingface.co/datasets/facebook/winoground)
for the account authenticated with `hf auth login`, plus Arial Bold for Mind2Web.
Use `--data DecisionIndex_Text DecisionIndex_Vision` when running the two public
tracks. Valen's native image budget is set automatically; see the [protocols](benchmarks.md).

## Download a model and evaluate

```bash
hf download Valen-Team/Valen-0.8B \
  --local-dir "$JEVALKIT_DATA_ROOT/models/Valen-0.8B"

python -m jevalkit run --model Valen-0.8B --data JevBench \
  --option model="$JEVALKIT_DATA_ROOT/models/Valen-0.8B" \
  --output "$JEVALKIT_DATA_ROOT/results"
```

Use `list models` / `list benchmarks` to inspect registered names. Multiple
benchmark names may follow `--data`. Add `--limit 10` for a smoke test or `--resume`
to continue an interrupted run with unchanged data, weights and settings.
`verify BENCHMARK --media` checks the complete TSV and referenced media.

Each result retains `predictions.jsonl`, `manifest.json` and `metrics.json`.
Re-score saved answers without loading weights with `python -m jevalkit score RUN_DIR`.

## APIs and author servers

Configure the service's credentials or endpoint in your environment, then select
its registered model. For example, OpenAI Decisions reads `OPENAI_API_KEY`:

```bash
python -m jevalkit run --model OpenAI-Decisions --data JevBench \
  --output "$JEVALKIT_DATA_ROOT/results"
```

Endpoint variables and author SDK paths are listed in the [model guide](models.md).
[Managed services](../examples/served-model.json) can start and stop a local author
server around evaluation; [hosted API examples](../examples/hosted-apis.json)
use credentials from the environment.

## Matrices and parallel runs

```bash
python -m jevalkit matrix examples/decision-benchmarks.json
```

Each matrix row can select its own `python`, checkpoint `options` and output path.
Rows run as isolated subprocesses; an independent row continues if another fails.

To spread one evaluation across workers, use the same `--world-size N` with a
unique `--rank` from `0` to `N-1` per worker; matrix rows accept `world_size` and
`rank` too. After every shard finishes, merge into a new empty directory:

```bash
python -m jevalkit merge /path/to/shard-000 /path/to/shard-001 \
  --model Pplx-Decider --data DecisionIndex \
  --output "$JEVALKIT_DATA_ROOT/results/merged/Pplx-Decider/DecisionIndex"
```

The merger checks source, runtime, checkpoint hashes and exact request coverage,
then recomputes global metrics. Partial shards never enter the primary leaderboard.
Use unsharded rows for managed HTTP services, whose manifests include worker paths.

## Results

```bash
python -m jevalkit leaderboard "$JEVALKIT_DATA_ROOT/results" --output "$JEVALKIT_DATA_ROOT/leaderboard"
```

[Open the leaderboard](https://huggingface.co/spaces/Valen-Team/JevalKit-Leaderboard)
or browse the generated Markdown, TSV and JSON. Compare matching dataset hashes
and protocols; latency also requires matching hardware. The latest eligible full
run is selected for each model and benchmark, rather than the best repeated score.
