"""Fastbite support routing: route each message to a queue and set safety and compensation flags.

Questions and decide() are the support-router recipe (valen-team.github.io static/cookbook/code/
skill_support_router.py at f9359e10561970c487835e3824effba94311c282), which a coding agent built
from the brief and dev examples only. cases.json is the 60-message hidden test set of Valen-Team/skills
evals/tasks/delivery-support/test.json at e70877b6a9425e90328c265bfedac8ef40527173, labeled by the
Valen team under the business rules in that task's labeling.md. Both are Apache-2.0.
"""

from pathlib import Path

from .. import action, source_revision
from ...io import load_json, sha256

CONTEXT = ("In-app support message from a customer of Fastbite, a food-delivery app in Singapore. "
           "Customers write in English or Chinese.")

QUEUE_CRITERIA = {
    "late_order": "The order has not arrived yet or arrived late, and the wait is the complaint or question",
    "missing_or_wrong_item": "Something ordered is missing, a different item came, or the order never arrived although marked delivered",
    "food_quality": "The food arrived but is cold, spilled, spoiled, undercooked, contains something it should not, or tastes wrong",
    "payment_refund": "Charges, double charges, a refund already promised, payment methods, or a voucher that failed at checkout",
    "account_access": "Login, verification codes or OTP, a locked or hacked account, changing phone number or email",
    "none": ("None of these: compliments or feedback about a restaurant or rider with nothing to fix, partnership or job "
             "enquiries, deleting an account, dangerous rider behaviour alone, greetings or messages with no clear request"),
}

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which problem does the customer's `message` report?",
        "criteria": QUEUE_CRITERIA,
    },
    "safety": {
        "type": "noul",
        "instructions": "The `message` reports a health or safety problem.",
        "criteria": {
            "true": "An allergic reaction, feeling sick after eating, a foreign object such as hair, plastic or metal in the food, raw or spoiled food, or a rider behaving dangerously",
            "false": "Food that is only cold, spilled, late, missing or wrong, a payment or account issue, or no problem",
        },
    },
    "compensation": {
        "type": "noul",
        "instructions": "The customer asks for money back, credit or a voucher.",
        "criteria": {
            "true": "The customer asks for a refund, money back, credit or a voucher, or chases a refund already promised",
            "false": "The customer only reports a problem, asks a question or asks for something else, without asking for money, credit or a voucher",
        },
    },
    "multi": {
        "type": "noul",
        "instructions": ("The `message` reports problems of two or more of these kinds: a late order, a missing or wrong item, "
                         "poor food quality, a payment problem, an account access problem."),
        "criteria": {
            "true": "Two or more of those kinds, such as a late order and a missing item, or a double charge and a cold meal",
            "false": ("Only one of those kinds, even if described in several sentences, with several affected items, with a refund "
                      "request for it, or alongside complaints about rudeness or remarks about the rider or restaurant; or none of them"),
        },
    },
    "delete_account": {
        "type": "noul",
        "instructions": "The customer wants to delete or permanently close their Fastbite account.",
        "criteria": {
            "true": "The customer asks to delete, close or deactivate their account for good",
            "false": "The customer wants to keep using the account, or the message is about something else",
        },
    },
    "describes_event": {
        "type": "noul",
        "instructions": "The `message` describes something specific that happened or is happening.",
        "criteria": {
            "true": "The customer tells about a specific event, situation or problem",
            "false": "Only a greeting, a single word, punctuation, or a message too short to say anything specific",
        },
    },
}

MULTI_THRESHOLD = 0.5
MULTI_BAND = 0.3
QUEUE_SPLIT_GAP = 0.2
SAFETY_THRESHOLD = 0.8
DELETE_ACCOUNT_THRESHOLD = 0.5
DESCRIBES_EVENT_THRESHOLD = 0.5
COMPENSATION_THRESHOLD = 0.5


def build_state(item):
    return {"context": CONTEXT, "order_status": item.get("order_status", "none"), "message": item.get("message", "")}


def decide(answers, message=""):
    queue = answers["queue"]["choice"]
    multi = answers["multi"]["noul"]
    safety = (answers["safety"]["noul"] >= SAFETY_THRESHOLD
              and answers["describes_event"]["noul"] >= DESCRIBES_EVENT_THRESHOLD
              and any(ch.isalnum() for ch in message))
    compensation = answers["compensation"]["noul"] >= COMPENSATION_THRESHOLD
    top, second = sorted(answers["queue"]["probabilities"].values(), reverse=True)[:2]
    split = multi >= MULTI_BAND and top - second < QUEUE_SPLIT_GAP
    deleting = answers["delete_account"]["noul"] >= DELETE_ACCOUNT_THRESHOLD
    if queue == "none" or multi >= MULTI_THRESHOLD or split or deleting:
        queue = "human_review"
    return {"queue": queue, "safety": bool(safety), "compensation": bool(compensation)}


def as_decision(route):
    """The queue is the primary action; each raised flag is one more action."""
    primary = action("route", {"queue": route["queue"]})
    return {"actions": [primary] + [action(flag) for flag in ("safety", "compensation") if route[flag]],
            "primary_action": primary}


class Workflow:
    name, title, default_reference = "delivery_support", "Delivery support routing", "human"
    policies = ["fastbite"]

    def __init__(self, revision=None, root=None):
        path = Path(__file__).with_name("cases.json")
        self.items = {item["id"]: item for item in load_json(path)}
        self.references = {"human": {cid: {"fastbite": as_decision(item["expected"])} for cid, item in self.items.items()}}
        self.dataset = {"source": "https://github.com/Valen-Team/skills/tree/main/evals/tasks/delivery-support",
                        "revision": "e70877b6a9425e90328c265bfedac8ef40527173", "cases_sha256": sha256(path),
                        "license": "apache-2.0", "labels": "human, Valen team"}
        self.revision = source_revision(Path(__file__).parent)

    def cases(self):
        return list(self.items)

    def execute(self, case_id, session):
        item = self.items[case_id]
        answers = session.ask(build_state(item), QUESTIONS)["answers"]
        route = decide(answers, item["message"])
        return {"decisions": {"fastbite": {**as_decision(route), "outputs": route}}, "details": {}}
