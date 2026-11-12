"""KnowLineModel adapter for the published KnowLine System One server.

Each KnowLine model repo ships serve_knowline.sh, which starts the engine and the
/v1/systemone front end with the settings of the model's Decision Index run.
"""
import copy

from ._http import HttpDecisionModel
from ..schema import Unsupported, is_message_state


class KnowLineModel(HttpDecisionModel):
    """Text decisions through the model repo's own /v1/systemone server.

    Structured states are sent unchanged, so the server's renderer sees the original
    fields. Requests with images or video are recorded as unsupported.
    """

    modalities = frozenset({"text"})

    def predict(self, request, *, media_root=None, assets=None):
        state = request["state"]
        if is_message_state(state):
            for message in state["messages"]:
                content = message.get("content")
                if isinstance(content, list) and any(part.get("type") != "text" for part in content):
                    raise Unsupported("This KnowLine adapter supports text states only")
        payload = {"model": self.model, "state": copy.deepcopy(state), "questions": copy.deepcopy(request["questions"])}
        return self.post(payload)
