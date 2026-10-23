"""Canonical Choice/Noul/Score requests. Gold labels never enter inference."""
import copy
import math
import re


class Unsupported(ValueError):
    """The selected release cannot represent this input without changing it."""


def is_capacity_error(message):
    """Only explicit author-runtime rejections, never arbitrary inference failures."""
    return (("above " in message and "truncation is forbidden" in message)
            or "exceed max_length=" in message or "Reserved tokenizer control token" in message
            or message.startswith("Question branch exceeds the ") and
            message.endswith("-token limit; no input was truncated.")
            or re.fullmatch(r"question .+: \d+ tokens is over the limit", message) is not None
            or re.fullmatch(r"score needs 2\.\.\d+ levels", message) is not None
            or re.fullmatch(r"question .+: only \d+ of its \d+ option markers fit in max_len=\d+ with "
                            r"head_max_len=\d+ spent on the question; lower head_max_len, raise max_len, "
                            r"or use fewer options", message) is not None
            or message.startswith("request has ") and "tokens, exceeding " in message
            and "reduce input size explicitly" in message)


def candidates(question):
    kind = question.get("type")
    if kind == "noul":
        criteria = question.get("criteria") or {}
        return {"false": criteria.get("false", "False"), "true": criteria.get("true", "True")}
    if kind == "choice":
        return question["criteria"]
    if kind == "score":
        return {str(i): description for i, description in enumerate(question["criteria"])}
    raise ValueError(f"Unknown question type: {kind}")


def distribution(value, keys, tolerance=0.01):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Probability keys must match every supplied candidate")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in value.values()):
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    total = sum(value.values())
    difference = abs(total - 1)
    at_boundary = math.isclose(difference, tolerance, rel_tol=0, abs_tol=8 * math.ulp(1.0))
    if total <= 0 or difference > tolerance and not at_boundary:
        raise ValueError(f"Probabilities sum to {total}, not 1")
    return {key: value[key] / total for key in keys}


def target_probabilities(question, target):
    if isinstance(target, dict) and target == {"unscored": True}:
        return None
    keys = list(candidates(question))
    if isinstance(target, dict) and "probabilities" in target:
        return distribution(target["probabilities"], keys, tolerance=1e-6)
    if isinstance(target, dict):
        value = target.get("label")
    else:
        value = target
    if isinstance(value, bool):
        value = "true" if value else "false"
    value = {"yes": "true", "no": "false"}.get(str(value), str(value)) if question["type"] == "noul" else str(value)
    if value not in keys:
        raise ValueError(f"Target {value!r} is not a candidate")
    return {key: float(key == value) for key in keys}


def target_label(question, target):
    """An explicit modal label may coexist with a soft probability reference."""
    if isinstance(target, dict) and target.get("label") is not None:
        # Reuse the strict label conversion independently of soft probabilities.
        truth = target_probabilities(question, {"label": target["label"]})
    else:
        truth = target_probabilities(question, target)
    return max(truth, key=truth.get) if truth is not None and max(truth.values()) == 1 else None


def validate_record(record):
    if not isinstance(record, dict) or not str(record.get("index", "")):
        raise ValueError("Record requires a nonempty index")
    request = record["request"]
    if set(request) != {"state", "questions"}:
        raise ValueError("Request must contain only state and questions")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Record requires named questions")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise ValueError("Question IDs must be nonempty strings")
        if set(question) - {"type", "instructions", "criteria", "labels"}:
            raise ValueError("Unexpected inference question fields")
        if not isinstance(question.get("instructions"), str):
            raise ValueError("Question requires instructions")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2
                                  or any(not isinstance(k, str) for k in criteria)):
            raise ValueError("Choice requires at least two named criteria")
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            raise ValueError("Score requires an ordered list of at least two levels")
        if kind == "noul" and criteria is not None and (not isinstance(criteria, dict)
                                                       or criteria and set(criteria) != {"true", "false"}):
            raise ValueError("Noul criteria must be true/false")
        candidates(question)
    if set(record.get("targets", {})) != set(questions):
        raise ValueError("Every evaluation question needs exactly one target")
    for qid, question in questions.items():
        target = record["targets"][qid]
        target_probabilities(question, target)
        target_label(question, target)
        if isinstance(target, dict) and "ordinal_values" in target:
            values = target["ordinal_values"]
            if question["type"] != "score" or not isinstance(values, list) or len(values) != len(candidates(question)):
                raise ValueError("Ordinal values must match every Score level")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Ordinal values must be finite numbers")
            if any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError("Ordinal values must strictly increase")
    if not isinstance(record.get("meta", {}), dict) or not isinstance(record.get("assets", []), list):
        raise ValueError("Invalid record metadata/assets")
    for qid, order in record.get("meta", {}).get("candidate_order", {}).items():
        if qid not in questions or len(order) != len(set(order)) or set(order) != set(candidates(questions[qid])):
            raise ValueError("Scoring candidate order must cover the complete candidate set")
    return record


def is_message_state(state):
    """Recognize the explicit chat envelope, not business records named messages."""
    return (isinstance(state, dict) and set(state) == {"messages"}
            and isinstance(state["messages"], list)
            and all(isinstance(message, dict) and isinstance(message.get("role"), str)
                    and "content" in message for message in state["messages"]))


def inference_request(record):
    """A copy prevents runtimes from mutating labels or subsequent benchmark inputs."""
    request = copy.deepcopy(record["request"])
    state = request["state"]
    if isinstance(state, dict) and "messages" in state and not is_message_state(state):
        from .io import dumps
        request["state"] = dumps(state)
    return request


def answer_probabilities(question, answer):
    if not isinstance(answer, dict) or answer.get("type") != question["type"]:
        raise ValueError("Native answer type does not match the question")
    if question["type"] == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid Noul probability")
        return {"false": 1 - p, "true": p}
    return distribution(answer.get("probabilities"), candidates(question))


def native_answer(question, probabilities):
    probabilities = distribution(probabilities, candidates(question))
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities["true"]}
    answer = {"type": kind, "probabilities": probabilities}
    if kind == "choice":
        answer["choice"] = max(probabilities, key=probabilities.get)
    else:
        answer["score"] = sum(int(k) * p for k, p in probabilities.items())
    return answer
