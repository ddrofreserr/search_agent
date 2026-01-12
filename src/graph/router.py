from typing import Literal
from src.graph.state import AgentState


def route_after_guard(state: AgentState) -> Literal["blocked", "ok"]:
    return "blocked" if state.get("guard_blocked") else "ok"


def route_after_handle_approval(state: AgentState) -> str:
    if state.get("approved") is True:
        return "approved"
    return "revise"
