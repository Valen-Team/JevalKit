"""Probability metrics with explicit coverage and hard/soft label denominators."""

import math
from collections import defaultdict

from .io import dumps
from .pricing import cost_usd, usage_tokens
from .schema import (
    answer_probabilities, candidates, is_capacity_error, is_message_state, target_label, target_probabilities,
)

SLICES = ("type", "subset", "domain", "modality", "script", "state_size", "candidates")


def question_metrics(question, target, probabilities, candidate_order=None):
    from .schema import target_label, target_probabilities

    truth = target_probabilities(question, target)
    if truth is None:
        return {"hard": False, "scorable": False}
    gold = target_label(question, target)
    hard = gold is not None
    # JevBench's native rule: lexical tie break, including ordinal questions.
    predicted = (
        max(candidate_order, key=probabilities.get)
        if candidate_order is not None
        else min(probabilities, key=lambda k: (-probabilities[k], k))
    )
    result = {
        "hard": hard,
        "correct": predicted == gold if hard else None,
        "predicted": predicted,
        "confidence": probabilities[predicted],
        "nll": -sum(t * math.log(max(probabilities[k], 1e-12)) for k, t in truth.items()),
        "brier": sum((probabilities[k] - t) ** 2 for k, t in truth.items()),
    }
    result["kl"] = sum(t * math.log(t / max(probabilities[k], 1e-12)) for k, t in truth.items() if t)
    if hard:
        result["hard_nll"] = -math.log(max(probabilities[gold], 1e-12))
        result["hard_brier"] = sum((probabilities[k] - float(k == gold)) ** 2 for k in truth)
    if question["type"] == "score":
        keys = list(truth)
        levels = (
            target.get("ordinal_values", [int(k) for k in keys])
            if isinstance(target, dict)
            else [int(k) for k in keys]
        )
        values = dict(zip(keys, levels))
        actual = sum(values[k] * p for k, p in truth.items())
        estimate = sum(values[k] * p for k, p in probabilities.items())
        cumulative = 0.0
        rps = 0.0
        for key in keys[:-1]:
            cumulative += probabilities[key] - truth[key]
            rps += cumulative**2
        result.update(mae=abs(actual - estimate), rps=rps / (len(keys) - 1))
    return result


def summarize(rows):
    """Failed and unsupported hard questions remain in the accuracy denominator."""
    hard = [r for r in rows if r["hard"]]
    valid = [r for r in rows if r["status"] == "ok"]
    calibrated = [r for r in valid if r["hard"]]
    out = {
        "questions": len(rows),
        "hard_questions": len(hard),
        "valid": len(valid),
        "failed": sum(r["status"] == "error" for r in rows),
        "unsupported": sum(r["status"] == "unsupported" for r in rows),
        "refused": sum(r["status"] == "refusal" for r in rows),
        "coverage": len(valid) / len(rows) if rows else 0,
        "accuracy": sum(bool(r.get("correct")) for r in hard) / len(hard) if hard else None,
        "probability_metric_questions": sum("nll" in r for r in valid),
        "ece_questions": len(calibrated),
    }
    for metric in ("nll", "brier", "kl", "hard_nll", "hard_brier", "mae", "rps"):
        values = [r[metric] for r in valid if metric in r]
        out[metric] = sum(values) / len(values) if values else None
    ece = 0.0
    for i in range(15):
        bucket = [r for r in calibrated if min(int(r["confidence"] * 15), 14) == i]
        if bucket:
            ece += abs(sum(r["confidence"] - bool(r["correct"]) for r in bucket)) / len(calibrated)
    out["ece"] = ece if calibrated else None
    latency = sorted(r["latency_s"] for r in rows if "latency_s" in r)
    out["request_latency_mean_s"] = sum(latency) / len(latency) if latency else None
    # Usage and cost belong to requests; a total is unknown if any request lacks it.
    requests = [r for r in rows if "latency_s" in r]
    for name in ("input_tokens", "output_tokens", "cost_usd"):
        values = [r.get(name) for r in requests]
        out[name + "_total"] = sum(values) if values and None not in values else None
    out["cost_usd_per_request"] = out["cost_usd_total"] / len(requests) if out["cost_usd_total"] is not None else None
    return out


def state_text(state):
    if isinstance(state, str):
        return state
    if is_message_state(state):
        return "\n".join(
            message["content"] if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"] if part.get("type") == "text")
            for message in state["messages"]
        )
    return dumps(state)


def script(text):
    if any("\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af" for c in text):
        return "cjk"
    return "latin" if any("a" <= c.lower() <= "z" for c in text) else "other"


def bucket(value, edges, labels):
    return next((label for edge, label in zip(edges, labels) if value < edge), labels[-1])


def slices(record, question):
    """Input properties that every benchmark shares, for finding where a model is weak."""
    text = state_text(record["request"]["state"])
    return {
        "script": script(text),
        "state_size": bucket(len(text), (1000, 4000, 16000), ("<1k", "1k-4k", "4k-16k", "16k+")),
        "candidates": bucket(len(candidates(question)), (3, 5, 9), ("2", "3-4", "5-8", "9+")),
    }


def grouped_report(rows):
    groups = {}
    for name in SLICES:
        buckets = defaultdict(list)
        for row in rows:
            buckets[str(row.get(name, "unknown"))].append(row)
        groups[name] = {key: summarize(value) for key, value in sorted(buckets.items())}
    return {"overall": summarize(rows), "groups": groups}


def score_predictions(dataset, predictions, rates=None):
    rows = []
    by_id = {p["index"]: p for p in predictions}
    for record in dataset:
        if record["index"] not in by_id:
            continue
        prediction = by_id[record["index"]]
        for qid, question in record["request"]["questions"].items():
            truth = target_probabilities(question, record["targets"][qid])
            row = {
                "index": record["index"],
                "qid": qid,
                "type": question["type"],
                "hard": target_label(question, record["targets"][qid]) is not None,
                "scorable": truth is not None,
                "status": prediction["status"],
                **{k: record["meta"].get(k, "unknown") for k in ("subset", "domain", "modality")},
                **slices(record, question),
            }
            if prediction["status"] == "error" and is_capacity_error(prediction.get("error", "")):
                row["status"] = "unsupported"
            # Latency belongs to shared requests, not to every question in the batch.
            if qid == next(iter(record["request"]["questions"])):
                row["latency_s"] = prediction["latency_s"]
                # A failed request keeps the usage its provider billed, when reported.
                tokens = usage_tokens(prediction.get("response") or prediction)
                row.update(input_tokens=tokens[0], output_tokens=tokens[1], cost_usd=cost_usd(rates, *tokens))
            if prediction["status"] == "ok":
                try:
                    answer = prediction["response"]["answers"][qid]
                    if not isinstance(answer, dict):
                        raise ValueError("Native answer must be an object")
                    if answer.get("type") == "refusal":
                        row["status"] = "refusal"
                    elif answer.get("error") in {"max_length_exceeded", "unsupported_question_type"}:
                        row.update(status="unsupported", error=answer["error"])
                    else:
                        probabilities = answer_probabilities(question, answer)
                        order = record["meta"].get("candidate_order", {}).get(qid)
                        row.update(
                            question_metrics(question, record["targets"][qid], probabilities, order),
                            probabilities=probabilities,
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    row.update(status="error", error=str(exc))
            rows.append(row)
    return rows
