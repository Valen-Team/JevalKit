# From calibrated probabilities to a review policy

A useful report answers two different questions: does a stated probability agree with observed correctness, and how much work can be accepted at a tolerable error rate? Calibration alone does not establish that a model ranks its mistakes well. Temperature scaling can improve NLL while ECE or selective performance gets worse.

The offline tools below need only Python's standard library. They never fetch weights, datasets or media. Existing benchmark/official scoring remains unchanged: the reliability report is a separate artifact, not a new leaderboard score.

## Reuse a retained run

First split by customer, conversation, media identity or other source entity, independently of training. If reusing an already evaluated public benchmark, disclose the study as retrospective and freeze the partition seed before examining results.

```bash
jeval reliability-export output/Valen-0.8B/Banking77 \
  --root /data/JevalKit --role unassigned \
  --protocol recipe.json --group-namespace banking77-utterances-v1 \
  --group-by normalized-state --output raw.json

jeval reliability split raw.json --seed my-frozen-seed --output cache
jeval reliability pipeline \
  --calibration-cache cache/calibration.json \
  --policy-cache cache/policy.json --test-cache cache/test.json \
  --output evidence
```

`--group-by meta` (default) requires explicit `meta.group_id`. `normalized-state` groups exact case/whitespace-normalized text; it cannot certify that near duplicates or related customer records are absent. Use real source-entity IDs when available. The namespace must remain the same across all three splits and models.

For three already independent runs, export each with `--role calibration`, `policy`, or `test` directly. Export requires complete selected-request coverage, verifies the retained dataset hash, preserves failures, and refuses already calibrated runs. It does not silently choose a hard label for a soft target. Positive saved probabilities can be replayed as `softmax(log(p)/T)`; rounded zero probabilities are rejected because lost logits cannot be recovered. Collect original logits through Valen's `python -m valen.reliability_collect` when necessary.

## Fit, freeze, test

```bash
jeval reliability fit cache/calibration.json --output temperature.json
jeval reliability select cache/policy.json --calibration temperature.json \
  --max-error 0.05 --min-accepted 30 --output policy.json
jeval reliability report cache/test.json --calibration temperature.json \
  --policy policy.json --output evidence
```

Fitting minimizes mean NLL on valid calibration questions, with positive scalar temperature bounded to `[0.01,100]`; `--per-type` fits separate Choice/Score/Noul temperatures and refuses missing task temperatures at replay. The default is a single scalar per model. Failed questions are excluded from fitting probability metrics but retained in operational coverage and hard-label accuracy denominators.

Selection maximizes validation coverage among thresholds whose pointwise Wilson 95% upper accepted-error bound is no larger than the requested target, with at least 30 accepted questions. If no threshold qualifies, `threshold=null` means review everything. Equal top probabilities are accepted together. The test threshold is never retuned. This heuristic is **not a formal error guarantee**: threshold search, correlated questions and deployment shift are not covered by a pointwise interval. Inspect actual test errors and counts.

`pipeline` runs all three stages, by default freezing rules for validation targets 1%, 2%, 5%, 10%. Use repeated `--max-error` to specify other targets before evaluating test. It writes:

- `calibration.json`: temperature, objective, bounded search and fitted source fingerprints.
- `policy-0.json` etc.: validation-only thresholds, calibration hash, validation counts and source-group fingerprints.
- `report.json`: before/after NLL, multiclass-sum Brier, ECE, accuracy, Score MAE/RPS, failure counts, reliability bins and risk-coverage points.
- `bins.csv`, `risk_coverage.csv`, `reliability.svg`, `risk-coverage.svg`: tables and lightweight standalone charts.

Reports group by task type, modality, domain, language and candidate count. `same_as: overall` reuses top-level before/after metrics when a group contains the entire sample. Empty bins have null accuracy, not fabricated zero accuracy. `probability_one_questions` counts hard-label predictions with top probability exactly 1.

ECE uses 15 equal-width bins of **maximum candidate probability**, not Valen/Jev's API concentration `confidence`. Pure soft labels contribute NLL/Brier and Score metrics, but not ECE/accuracy/risk; an explicit `gold` may supply a separately known label. JevBench's official 10-bin ECE stays separate. Wilson intervals are pointwise and assume independent questions. Coverage is accepted / all eligible hard-label questions, including failures. Error rate is errors / accepted, null when nothing is accepted. AURC integrates right-step risk only over attainable coverage, so compare it only with matching coverage/failure scope.

## Load fitted parameters in the native Valen adapter

```bash
jeval run --model Valen-0.8B --data Banking77 --output calibrated-run \
  --option revision=PINNED_REVISION \
  --option calibration=evidence/calibration.json \
  --option calibration_protocol=recipe.json
```

The adapter verifies model, revision, backend, precision, recipe fingerprint and every bound artifact hash, and records the calibration fingerprint. Extra runtime file hashes are allowed; missing or changed bound files are rejected. Request-body temperature overrides are refused while calibration is active. Raw runs keep T=1 and existing behavior.

A recipe JSON records the application/prompt/candidate/label/media/grouping versions. It is a declaration that must match the real inference setup, not an automatic semantic prompt-equivalence checker. Refit when these inputs or the deployment population change. Historical published BANKING77 parameters are not universal defaults for other tasks or backends.

## Cache protocol v1

A cache is a JSON object with `schema_version: 1`, `role` (`unassigned`, `calibration`, `policy`, `test`), `identity`, `dataset`, `group_namespace`, and `rows`.

`identity` contains nonempty `model`, `revision`, `backend`, `precision`, a lowercase `protocol_sha256`, and nonempty `artifacts_sha256` mapping filenames to SHA-256. Rows require unique string `id`, stable string `group_id`, `type`, at least two ordered string `candidates`, normalized numeric `target`, and explicit `status` (`ok`, `error`, `unsupported`, `refusal`). Valid rows additionally carry finite `logits` or strictly positive normalized `probabilities`, in candidate order. Optional `gold` is an explicit hard-label candidate. Optional `modality`, `domain` and `language` drive groups. No prompt text is required for replay.

Source groups are hashed in saved artifacts; overlap checks reject leakage between fit, policy and test, including renamed row IDs. These checks cannot certify absence of training contamination or semantic duplicates: disclose those limits separately. The v1 standard-library implementation is mirrored byte-for-byte in Valen's `valen/evaluation/calibration.py`; maintain both protocol suites when changing it.


After downloading a report, audit ECE, accuracy, NLL/Brier means and coverage arithmetic and regenerate figures without weights or private per-question caches. This checks internal consistency; authenticating model/data inputs still requires the original retained predictions and their hashes.

```bash
jeval reliability audit report.json --output checked-figures
```
