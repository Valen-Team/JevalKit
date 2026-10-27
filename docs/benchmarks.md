# Benchmark protocols

**English** · [简体中文](benchmarks_zh-CN.md)

[Project overview](../README.md) · [Running evaluations](quickstart.md) · [Model integrations](models.md)

Eight Jev-related benchmark families are implemented, exposed through twelve TSV views. The
Image/Video views and Decision Index Text/Vision tracks belong to their respective families.

| Family | Evaluation data | Scoring |
|---|---|---|
| VisualDecisionBench | Full released `eval.jsonl`, 1,696 shared states, 9,143 questions: 2,000 image and 7,143 video | Hard-label accuracy; soft-label NLL/Brier; ordinal MAE/RPS; modality/type/domain breakdowns |
| JevBench | 231 frozen public tasks: 48 easy, 72 original, 111 hard | Vendored official task scorer, calibration and family/paraphrase reports |
| Decision Index | 0.4.1 public Text: 140,178 requests; Vision: 12,210 questions across 11 benchmarks; legacy 0.3 retained | Official chance-corrected weighted public indices |
| Typed-Decisions | Pinned `all/test`: 400 shared states, 2,000 questions | Explicit modal-label accuracy; soft-reference KL, Brier and cross-entropy |
| DecisionBench | Pinned standard `eval`: 23,900 rows, three primitives | All-row accuracy; candidate-aligned NLL and calibration; original ordinal values |
| Procedural Typed Decisions | Pinned `all/test`: 18,000 states, 57,016 questions | Exact rule judgments; pure posterior questions use probability error without invented hard labels |
| zh-decision-bench | Pinned v0.2 public validation: 378 states, 467 questions | Chinese classification/rating; right-closed 15-bin reference ECE |
| JevAlt-Bench | Pinned English/Turkish/German test files: 3,116 states, 8,434 questions | Modal accuracy; hard and soft probability scores; language slices and equal-mass reference ECE |

Each [benchmark directory](../jevalkit/datasets/) contains preparation, scoring
and a `manifest.json` pinning TSV hashes, counts and protocols. Download the released data
using the [usage guide](quickstart.md#download-data).

`JevBench` specifically means
[`fstandhartinger/jevbench`](https://github.com/fstandhartinger/jevbench), revision
`bb05a335bc809e61b20c0f745d25499a82b326fc`, used by Valen's public comparison.
Other projects named JevBench use different tasks and cannot share these scores.
The original task's custom true/false rubric is preserved for Noul inference.

Decision Index preserves question IDs, options, order, structured state, gold
labels, grouping, official exclusions and the frozen suite's source metadata.
Unlabeled diagnostic questions remain in the request and carry an explicit
`{"unscored": true}` target. They do not contribute to probability or accuracy
metrics. The official scorer is required for the headline weighted index;
generic field accuracy is a separate diagnostic, not a replacement.

`DecisionIndex_Text` and `DecisionIndex_Vision` target board release **0.4.1**.
The official kit still labels both public suites **0.3**: text uses the unchanged
37-benchmark index panel; vision uses the eleven published benchmark weights.
Both scores cover the public part. `DecisionIndex` retains the legacy text protocol.

Vision TSVs preserve image order, original question/options, payload and image hashes,
and the engine-specific prompt metadata. Valen runs automatically use the board's
**1,638,400-pixel** native processor budget; a lower budget is rejected for this track.
Original source images remain unchanged. Native decision heads read the typed
questions directly; generative assistant readout prefixes remain source metadata.

Rebuild vision from the [official pinned sources](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/docs/vision.md),
with Winoground access approved on Hugging Face and Arial Bold supplied through
`DECISION_INDEX_FONT`; Pillow 12.3.0 reproduces the reference image hashes.
Use `python -m jevalkit prepare DecisionIndex_Vision --source /path/to/verified/vision`
for an existing suite. These source licenses require local reconstruction; the
toolkit does not grant permission to redistribute the resulting images or TSVs.

Inference never receives gold labels, rationales, evaluation IDs or source
metadata. Native runtimes retain their published temperatures and context limits.
Models which cannot represent a modality or question type report it as
unsupported. Their failures remain visible in coverage and accuracy.

All probabilities must be finite, nonnegative, within [0,1], with complete
candidate coverage. Generic metrics allow at most 0.01 rounding error in the
sum, recording the raw answer and normalizing only for generic metrics. The
official JevBench scorer retains its own published strict/rounded tolerances and
lexical argmax tie rule. Its official aggregates are retained separately.
The Decision Index integration also guards against floating-point noise at its
validator's 0.01 boundary. A temporary copy is used only for that check; original
answers are passed to the pinned official scorer, and the compatibility count is recorded.

NLL floors probabilities at 1e-12; multiclass Brier is the sum of squared errors;
ECE uses 15 equal-width confidence bins; ordinal RPS averages squared cumulative
errors over the first K−1 levels. Soft targets without an explicit classification label have no hard-label accuracy.
When a source provides both a label and a distribution, accuracy/ECE use the
label while proper probability scores use the complete distribution.
Per-request synchronized latency includes preprocessing and native inference,
excludes model loading/warmup, and is counted once per multi-question request.

Dataset/provider licenses and usage restrictions remain in force. The toolkit uses the released evaluation snapshots and their documented source
licenses; see [third-party terms](../THIRD_PARTY.md).

## Additional decision-suite protocols

The TSV snapshots preserve complete shared requests and the source's candidate
order. Hash checks reject changed data before inference.

- **Typed-Decisions:** decode JSON states, preserve all five questions in one
  request, and retain both explicit labels and normalized soft golds. Teacher
  annotations measure teacher agreement. General and workflow-finetuned models
  have different exposure; compare them with that distinction visible.
- **DecisionBench:** use the upstream System One projection: the first binary
  candidate maps to Noul `true`; Choice keeps original candidate IDs; Score
  sorts by original ordinal value and stores the inverse ID mapping. Scoring
  restores original candidate order for ties and original numeric values for
  ordinal errors. Compact fields are not used. The reference NLL floor is
  1e-15; canonical generic NLL retains its 1e-12 floor.
- **Procedural Typed Decisions:** preserve every task, difficulty and question.
  Pure probabilistic Noul targets remain soft: they contribute probability
  losses and posterior MAE, not fabricated hard accuracy or empirical ECE.
  The pinned 18-task release differs from older RSI evaluation snapshots.
- **zh-decision-bench:** read the published v0.2 payload (323 routing and 55
  business items), map Chinese Score labels to their supplied level positions,
  and retain the source's `true,false` tie order. The Hub card has stale count
  tables; verified payload counts are 378 items / 467 questions. This is its
  public validation set; no test-time temperature refit is performed.
- **JevAlt-Bench:** use file membership for the test split despite placeholder
  row fields saying `train`. The published input `abstain` flag adds an explicit
  `unknown` Choice candidate; gold labels never determine model input. This
  common interface is an explicitly tagged projection of the author's API
  abstention flag. Dictionary Score rubrics retain insertion order and label
  mappings. Extra upstream question fields are serialized into instructions.
  These interface choices must be matched before comparing author scores.

`metrics` contains common JevalKit measurements. For these five suites the
historically named `official` field contains a **JevalKit reference protocol
report**, identified by `scorer=jevalkit-decision-protocol-v1`; it does not imply
execution by an upstream harness. Reports include task/domain/language/difficulty
slices, label-only NLL/Brier, soft-target KL/Brier, and the applicable reference
ECE. JevBench and Decision Index continue to invoke their benchmark-owned scorers.

## Add a benchmark

Create `jevalkit/datasets/<benchmark>/` with `__init__.py` and `manifest.json`.
Implement `prepare(root=None, source=None, **options)` and, when required,
`evaluate(dataset, records, predictions, output)` in that package. Register the
package name, family and modality in `jevalkit/datasets/__init__.py`; the CLI and
runner dispatch through that entry. Use `_shared.commit_dataset` for validated TSVs
and add fixtures checking the source projection and scoring rules.
