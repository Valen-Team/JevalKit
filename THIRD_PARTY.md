# Sources and attribution

The repository structure is inspired by [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
No VLMEvalKit implementation is copied.

`jevalkit/datasets/jevbench/official/` contains the task schema and scoring/aggregation
modules from [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench),
revision `bb05a335bc809e61b20c0f745d25499a82b326fc`, obtained through Valen's
attributed vendor snapshot. Its MIT license is retained in that directory.
This is the 231-task public benchmark used by Valen, distinct from other
projects also named JevBench.

Decision Index scoring is delegated to the separately installed upstream kit,
pinned in `pyproject.toml`. Its public index does not include private evaluations.

Model inference is delegated to the authors' published packages or release
modules. Weights, model code and benchmark datasets retain their own licenses.
Dataset manifests record source repositories, revisions, splits and checksums.
Do not treat JevalKit's Apache-2.0 license as a relicensing of those assets.

`jevalkit/workflows/invoice_processing/upstream/` is `evals/invoice_processing/structured/`
from [typesafe-ai/WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals), revision
`0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb`, Apache-2.0 (license retained in that directory). Only its
`typesafe_sdk` import was changed, to `jevalkit.workflows.sdk_types`. `jevalkit/pricing.json` takes its rates
from the same repository's `core/pricing.json`. The workflow's cases and reference labels are downloaded from
[typesafeAI/evalsafe-invoice-processing](https://huggingface.co/datasets/typesafeAI/evalsafe-invoice-processing),
Apache-2.0, at the revision recorded in each run.

OpenAI Decisions uses the documented `/v1/decisions` schema:
https://developers.openai.com/api/reference/resources/decisions/methods/create
The adapter maps Noul to Predicate and preserves ordered Score levels.

Additional decision benchmarks are downloaded at pinned revisions, with source
file hashes retained in the dataset manifests. Their data is stored outside Git:

- [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): Apache-2.0.
- [DecisionBench](https://huggingface.co/datasets/Hanno-Labs/decision-bench): source-specific terms; the collection declares `other`.
- [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions): Apache-2.0, Damien Sileo/tasksource.
- [zh-decision-bench](https://github.com/CodyQin/zh-decision-bench): CC-BY-4.0 data; CodyQin's authored scenarios and Amazon MASSIVE Chinese routing records.
- [JevAlt-Bench](https://huggingface.co/datasets/mertkayacs/jevalt-bench): record-level mixed permissive terms, including Apache-2.0, CC0 and CC-BY-4.0 with original source attribution.

JevalKit's DecisionBench request projection follows the upstream System One
adapter at `Hanno-Labs/decision-bench@89aaa85827912f8975c1835348ead7436710f581`.
Reference scoring conventions were checked against that repository,
`mertkayacs/jevoss@0dbffa411fc4adfc1728113a5d34bc6adfb142cb` and
`CodyQin/zh-decision-bench@6769a45fc15c7d568450a9d34c977692a3f5faa6`.
These independently implemented protocol reports are identified separately from
the benchmark-owned scorers used for JevBench and Decision Index.
