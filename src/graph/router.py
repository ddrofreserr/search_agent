from typing import Literal
from src.graph.state import AgentState


def route_after_guard(state: AgentState) -> Literal["blocked", "ok"]:
    return "blocked" if state.get("guard_blocked") else "ok"


def route_after_handle_approval(state: AgentState) -> Literal["approved", "revise"]:
    """
    Router = часть планировщика/контроллера.

    Чем является:
    - Это НЕ узел, а функция, которая возвращает "ключ маршрута".
    - LangGraph использует этот ключ, чтобы выбрать нужное ребро (edge).

    Как работает:
    - если approved=True -> идём по ветке "approved" (к compose_answer)
    - иначе -> "revise" (возвращаемся к approval и спрашиваем заново)
    """
    if state.get("approved"):
        return "approved"
    return "revise"
