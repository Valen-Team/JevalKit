"""The typesafe_sdk question and answer types that vendored workflows use, over JevalKit's native dicts."""


def NoulCriteria(**outcomes):
    return dict(outcomes)


class Question:
    type = None

    def __init__(self, instructions, criteria=None):
        self.instructions, self.criteria = instructions, criteria

    def wire(self):
        """The canonical request question; undescribed Noul outcomes are left out."""
        criteria = self.criteria
        if self.type == "noul" and criteria is not None:
            criteria = {k: v for k, v in criteria.items() if v is not None} or None
        question = {"type": self.type, "instructions": self.instructions}
        if criteria is not None:
            question["criteria"] = criteria
        return question


class Noul(Question):
    type = "noul"


class Choice(Question):
    type = "choice"


class Score(Question):
    type = "score"


QuestionSet = dict


class NoulAnswer:
    type = "noul"

    def __init__(self, noul):
        self.noul = noul


class ChoiceAnswer:
    type = "choice"

    def __init__(self, choice, probabilities, confidence=None):
        self.choice, self.probabilities, self.confidence = choice, probabilities, confidence


class ScoreAnswer:
    type = "score"

    def __init__(self, score, probabilities, confidence=None):
        self.score, self.probabilities, self.confidence = score, probabilities, confidence


class Usage(dict):
    def model_dump(self, mode=None, exclude_none=False):
        return {k: v for k, v in self.items() if not (exclude_none and v is None)}


class SystemOneResponse:
    def __init__(self, answers, usage, model):
        self.answers, self.usage, self.model = answers, Usage(usage), model


def from_native(answers, usage, model):
    """Wrap native answers ({"type", "noul"} or {"type", "probabilities", ...}) as SDK answers."""
    wrapped = {}
    for qid, answer in answers.items():
        if answer["type"] == "noul":
            wrapped[qid] = NoulAnswer(answer["noul"])
        elif answer["type"] == "choice":
            probabilities = answer["probabilities"]
            wrapped[qid] = ChoiceAnswer(answer.get("choice") or max(probabilities, key=probabilities.get),
                                        probabilities, answer.get("confidence"))
        else:
            probabilities = answer["probabilities"]
            score = answer.get("score", sum(int(k) * p for k, p in probabilities.items()))
            wrapped[qid] = ScoreAnswer(score, probabilities, answer.get("confidence"))
    return SystemOneResponse(wrapped, usage, model)
