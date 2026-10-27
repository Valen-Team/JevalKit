"""Decision Index 0.4.1 text public track; its frozen suite edition remains 0.3."""
from ..decision_index import prepare as prepare_text, evaluate as evaluate_text


def prepare(root=None, source=None, edition="0.4.1"):
    if edition != "0.4.1":
        raise ValueError("DecisionIndex_Text supports board release 0.4.1; use DecisionIndex for legacy editions")
    return prepare_text(root, source, "0.3", leaderboard_version=edition)


def evaluate(dataset, records, predictions, output):
    report = evaluate_text(dataset, records, predictions, output)
    report.update(leaderboard_version="0.4.1", public_suite_edition="0.3", score_scope="public_text")
    return report
